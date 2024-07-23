import torch
from tu.utils.config import build_from_config
import os
from PIL import Image
from typing import List, Dict
import numpy as np
import torchvision.transforms.functional as TF
from langint.utils.dataset import imagenet_templates_small
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import Counter
from transformers import CLIPVisionModel
import kornia
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


placeholder_words_list = [f'mytoken{i}' for i in range(500)]


class Synthetic(torch.utils.data.Dataset):
    def __init__(self, data_root: str, num_placeholder_words: int):
        super().__init__()

        self.data_root = data_root.replace('_', " ")
        self.templates = imagenet_templates_small

        pipeline = GLIDEPipeline()
        ground_truth_words = data_root.split(',')
        placeholder_words = placeholder_words_list[:num_placeholder_words]
        assert len(placeholder_words) == num_placeholder_words, (placeholder_words, num_placeholder_words)
        assert len(placeholder_words) == 1 or len(ground_truth_words) == len(placeholder_words), (ground_truth_words, placeholder_words)
        if len(placeholder_words) == 1:
            placeholder_words = placeholder_words * len(ground_truth_words)
        images_all: List[torch.Tensor] = []
        ph_words_all: List[str] = []
        for ind in range(len(ground_truth_words)):
            gt_word = ground_truth_words[ind]
            ph_word = placeholder_words[ind]
            prompt = self.templates[0].format(gt_word)
            images = pipeline.sample(prompt).cpu()
            images_all.append(images)
            ph_words_all.extend([ph_word] * len(images))
        self.images: torch.Tensor = torch.cat(images_all)
        self.placeholder_words: List[str] = ph_words_all
        del pipeline

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # GLIDE expects range [-1, 1]
        image: torch.Tensor = self.images[item]
        if np.random.rand() < .5:
            image = TF.hflip(image)

        ph_word = self.placeholder_words[item]
        prompt = self.templates[np.random.choice(len(self.templates))].format(ph_word)

        return {'image': image, 'prompt': prompt}

def glide_sample_prompt(pipeline, prompt: str, num_repeats=4):
    images: torch.Tensor = pipeline.sample(prompt, batch_size=num_repeats).cpu()
    return images

def glide_sample_prompts(prompts: List[str], num_repeats=4) -> torch.Tensor:
    from langint.utils.glide import GLIDEPipeline
    # return (bs, 3, 64, 64) in pixel range [-1, 1]
    pipeline = GLIDEPipeline()
    images_all: List[torch.Tensor] = []
    for prompt in prompts:
        images = glide_sample_prompt(pipeline, prompt, num_repeats=num_repeats)
        images_all.append(images)
    del pipeline
    return torch.cat(images_all)

### Use DeepFloyd to generate images from prompts, and get answers from BLIP
def deepfloyd_sample_prompts(prompts: List[str], num_repeats=4, model=None, processor=None, blip_fruit_q=None, blip_color_q=None):
    from langint.utils.deepfloyd_no_diffusers import Pipeline
    pipeline = Pipeline()
    images_all: List[Image.Image] = []
    blip_common_fruits = []
    blip_common_colors = []

    ### Save to visualize the generated images
    img_save_path = os.path.join('/users/ljunyu/data/ljunyu/code/concept/langint-train-two/cache', 'deepfloyd_img')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    for prompt_idx, prompt in enumerate(prompts):
        images: List[Image.Image] = pipeline.dream(prompt, count=num_repeats)
        images_all.extend(images)

        if model is not None:
            blip_fruits = []
            blip_colors = [] # blip process the images here
            for image_i in range(len(images)):
                image = images[image_i]
                assert image.mode == 'RGB', image.mode

                image.save(os.path.join(img_save_path, str(prompt_idx) + '_' + str(image_i) + '.png'))

                inputs = processor(image, blip_fruit_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs)
                blip_fruits.append(processor.decode(blip_out[0], skip_special_tokens=True))

                inputs = processor(image, blip_color_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs)
                blip_colors.append(processor.decode(blip_out[0], skip_special_tokens=True))
            blip_fruit_counter = Counter(blip_fruits)
            blip_common_fruit = blip_fruit_counter.most_common(1)[0][0]
            blip_common_fruits.append(blip_common_fruit)


            blip_color_counter = Counter(blip_colors)
            blip_common_color = blip_color_counter.most_common(1)[0][0]
            blip_common_colors.append(blip_common_color)
    
    if model is not None:
        assert len(prompts) == len(blip_common_colors) == len(blip_common_fruits), (len(prompts), len(blip_common_colors), len(blip_common_fruits))

    del pipeline
    return torch.stack([TF.to_tensor(image) * 2 - 1 for image in images_all]), blip_common_colors, blip_common_fruits, images_all

### Used for held-out inference, load image samples and pass through BLIP to obtain lable
def load_samples_blip(prompts: List[str], num_repeats=4, model=None, processor=None, blip_fruit_q=None, blip_color_q=None):
    images_all: List[Image.Image] = []
    blip_common_fruits = []
    blip_common_colors = []

    samples_path = '/users/ljunyu/data/ljunyu/code/concept/langint-train-two/cache/deepfloyd_img'
    images_all = load_samples_as_PIL(samples_path)

    for prompt_idx, prompt in enumerate(prompts):
        images = images_all[prompt_idx*num_repeats : (prompt_idx+1)*num_repeats]

        if model is not None:
            blip_fruits = []
            blip_colors = [] # blip process the images here
            for image_i in range(len(images)):
                image = images[image_i]
                assert image.mode == 'RGB', image.mode

                inputs = processor(image, blip_fruit_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs)
                blip_fruits.append(processor.decode(blip_out[0], skip_special_tokens=True))

                inputs = processor(image, blip_color_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs)
                blip_colors.append(processor.decode(blip_out[0], skip_special_tokens=True))
            blip_fruit_counter = Counter(blip_fruits)
            blip_common_fruit = blip_fruit_counter.most_common(1)[0][0]
            blip_common_fruits.append(blip_common_fruit)


            blip_color_counter = Counter(blip_colors)
            blip_common_color = blip_color_counter.most_common(1)[0][0]
            blip_common_colors.append(blip_common_color)
    
    if model is not None:
        assert len(prompts) == len(blip_common_colors) == len(blip_common_fruits), (len(prompts), len(blip_common_colors), len(blip_common_fruits))

    return torch.stack([TF.to_tensor(image) * 2 - 1 for image in images_all]), blip_common_colors, blip_common_fruits, images_all


def load_samples_as_PIL(samples_path):
    # List all files in the folder
    files = [f for f in os.listdir(samples_path) if f.endswith('.png')]
    
    # Sort files based on the first 'xx' numerically
    files.sort(key=lambda x: int(x.split('_')[0]))
    
    # Load images using PIL and add them to a list
    images = []
    for filename in files:
        img_path = os.path.join(samples_path, filename)
        try:
            with Image.open(img_path) as img:
                images.append(img.copy())  # Use copy to avoid closing the file reference
        except IOError:
            print(f"Error opening image: {img_path}")
    
    return images

def cache_deepfloyd_samples(prompts: List[str], num_repeats=4) -> torch.Tensor:
    cache_dir = 'cache/deepfloyd'
    image_paths_all = []
    pipeline = None
    for prompt in prompts:
        cache_subdir = prompt.replace(" ", "_")
        os.makedirs(os.path.join(cache_dir, cache_subdir), exist_ok=True)
        image_paths = [os.path.join(cache_dir, cache_subdir, f"{ind:02d}.png") for ind in range(num_repeats)]
        if not all(os.path.exists(path) for path in image_paths):
            if pipeline is None:
                from langint.utils.deepfloyd_no_diffusers import Pipeline
                pipeline = Pipeline()
            images: List[Image.Image] = pipeline.dream(prompt, count=num_repeats)
            for ind in range(num_repeats):
                images[ind].save(image_paths[ind])
        image_paths_all.extend(image_paths)
    if pipeline is not None:
        del pipeline
    return image_paths_all

def load_deepfloyd_samples(prompts: List[str], num_repeats=4):
    logger.info('loading deepfloyd samples from cache...')
    image_paths = cache_deepfloyd_samples(prompts, num_repeats)
    images: List[Image.Image] = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
    return torch.stack([TF.to_tensor(image) * 2 - 1 for image in images])

class HookFunction:
    def __init__(self):
        self.layer_outputs = []

    def hook_layers(self, model):
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L206
        layer_counts = 0
        for layer in model.transformer.resblocks:
            layer_counts += 1
            assert layer.__class__.__name__ == 'ResidualAttentionBlock'
            layer.register_forward_hook(self.save_output)
        assert layer_counts > 0

    def save_output(self, module, input, output):
        self.layer_outputs.append(output.detach())

    def clear_outputs(self):
        self.layer_outputs = []

class SyntheticBiLevel(torch.utils.data.Dataset):
    def __init__(self, data_root: str,
                 templates: Dict,
                 num_data_per_prompt: int = 8, num_data_copies: int = 1, num_tokens_per_word: int = 1,
                 num_placeholder_words: int = 22, num_placeholder_groups: int = 2, shared_tokens=0): 
        
        assert shared_tokens in [0, 1], shared_tokens

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        logger.info('Check input variables:')
        logger.info(f'data_root: {data_root}')
        logger.info(f'templates: {templates}')
        logger.info(f'num_data_per_prompt: {num_data_per_prompt}')
        logger.info(f'num_data_copies: {num_data_copies}')
        logger.info(f'num_tokens_per_word: {num_tokens_per_word}')
        logger.info(f'num_placeholder_words: {num_placeholder_words}')
        logger.info(f'num_placeholder_groups: {num_placeholder_groups}')
        logger.info(f'shared_tokens: {shared_tokens}')
        logger.info('')
        
        ###################### Load BLIP models, generate images using DeepFloyd, get BLIP answers ######################
        logger.info('Check is initializing SyntheticBiLevel - BLIP')
        blip_path = '/users/ljunyu/data/ljunyu/code/concept/langint-train-two/cache/blip2-flan-t5-xl'
        # processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")
        processor = Blip2Processor.from_pretrained(blip_path)
        model = Blip2ForConditionalGeneration.from_pretrained(blip_path , load_in_8bit=True, device_map="auto")
        # we limit it to words which are known to have corresponding t5 embeddings which are one token long
        # blip_fruit_question = "Which of these is the fruit in the photo: cherry, apple, banana, mango, strawberry, pineapple, lemon, or raspberry?"
        # blip_color_question = "Which of these is the color of the fruit in the photo: red, blue, green, purple, yellow, orange, or black?"
        blip_fruit_question = "Which of these is the type of car in the photo: sports, van, muscle, wagon, race, truck, coupe, or limousine?"
        blip_color_question = "Which of these is the color of the car in the photo: red, blue, green, purple, yellow, orange, or black?"

        self.templates = build_from_config(templates)
        logger.info(f'Templates after build_from_config: {self.templates}')
        
        ground_truth_words = data_root.split(",")
        ground_truth_words = [word.replace('_', " ") for word in ground_truth_words]
        ground_truth_words = [word.split('-') for word in ground_truth_words] # [['apple', 'green'], ['apple', 'red'], ...]
        logger.info(f'ground_truth_words: {ground_truth_words}')
        assert len(ground_truth_words) == num_placeholder_words, (ground_truth_words, len(ground_truth_words), num_placeholder_words)
        ground_truth_prompt_args = [[] for i in range(num_placeholder_groups)]
        for split_word in ground_truth_words:
            assert len(split_word) == num_placeholder_groups
            for i in range(num_placeholder_groups):
                ground_truth_prompt_args[i].append(split_word[i])
                # now we have [apple, apple, banana, banana] and [green, red, green, yellow]
        logger.info(f'ground_truth_prompt_args: {ground_truth_prompt_args}')

        self.ground_truth_prompt_args = ground_truth_prompt_args
        self.unique_gt_words = ground_truth_words
        # print(self.unique_gt_words)
        # print(num_placeholder_groups)
        for gt_word in self.unique_gt_words:
            assert len(gt_word) == num_placeholder_groups

        unique_prompts = []
        for ind in range(num_placeholder_words):
            curr_prompt_words = []
            for ground_truth_prompt_arg in ground_truth_prompt_args:
                curr_prompt_words.append(ground_truth_prompt_arg[ind])
            prompt = self.templates[0].format(*curr_prompt_words)
            unique_prompts.append(prompt)
        logger.info(f'unique_prompts: {unique_prompts}')
        self.gt_prompts: List[str] = unique_prompts
        
        ### !!! Currently hardcoded to load pregenerated images (for heldout inference)
        # self.images, self.blip_colors, self.blip_fruits, pil_images = deepfloyd_sample_prompts(unique_prompts, num_repeats=num_data_per_prompt, model=model, processor=processor, blip_fruit_q=blip_fruit_question, blip_color_q=blip_color_question)   
        self.images, self.blip_colors, self.blip_fruits, pil_images = load_samples_blip(unique_prompts, num_repeats=num_data_per_prompt, model=model, processor=processor, blip_fruit_q=blip_fruit_question, blip_color_q=blip_color_question)
        
        # logger.info(f'After deepfloyd_sample_prompts, self.images: {self.images}')
        logger.info(f'After deepfloyd_sample_prompts, self.blip_colors: {self.blip_colors}')
        logger.info(f'After deepfloyd_sample_prompts, self.blip_fruits: {self.blip_fruits}')
        # logger.info(f'After deepfloyd_sample_prompts, self.pil_images: {self.pil_images}')
        
        # we limit it to words which are known to have corresponding t5 embeddings which are one token long
        ### Note: Need to change here to restrict BLIP answers
        new_blip_fruits = []
        for i in range(len(self.blip_fruits)):
            # to_append = 'fruit'
            # for x in ['cherry', 'apple', 'banana', 'mango', 'strawberry', 'pineapple', 'lemon', 'raspberry']:
            to_append = 'car'
            for x in ['sports', 'van', 'muscle', 'wagon', 'race', 'truck', 'coupe', 'limousine']:
                if x in self.blip_fruits[i]:
                    to_append = x
            new_blip_fruits.append(to_append)
        logger.info(f'new_blip_fruits: {new_blip_fruits}')

        new_blip_colors = []
        for i in range(len(self.blip_colors)):
            to_append = 'color'
            for x in ['red', 'blue', 'green', 'purple', 'yellow', 'orange', 'black']:
                if x in self.blip_colors[i]:
                    to_append = x
            new_blip_colors.append(to_append)
        logger.info(f'new_blip_colors: {new_blip_colors}')

        self.blip_fruits = new_blip_fruits
        self.blip_colors = new_blip_colors

        del processor
        del model
        torch.cuda.empty_cache()
        logger.info('')
        #########################################################################################################

        ###################### Load CLIP model and get CLIP features from generated images ######################
        #### Load local CLIP models (save time)
        clip_path = "/users/ljunyu/data/ljunyu/code/concept/deepfloyd/clip-vit-large-patch14"
        logger.info('Check is initializing CLIP - CLIP')
        # clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").requires_grad_(False)
        clip_vision = CLIPVisionModel.from_pretrained(clip_path).to("cuda").requires_grad_(False)
        def clip_preprocess(x):
            x = kornia.geometry.resize(
                x, (clip_vision.config.image_size, clip_vision.config.image_size), interpolation='bicubic', align_corners=True, antialias=False
            )
            x = (x + 1.) / 2.
            # renormalize according to clip
            x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
            return x

        ##### Generate CLIP features from the generated images
        preprocessed_images = self.images.to("cuda")
        preprocessed_images = clip_preprocess(preprocessed_images)

        run_count = 0
        self.clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                self.clip_features.append(result)
                run_count += 1

        self.clip_features = [torch.stack(clip_feature, dim=1).type(torch.float32).to("cpu") for clip_feature in self.clip_features]

        for i in range(len(self.clip_features)):
            assert self.clip_features[i].shape == (3, 12, 1024)
        assert len(self.clip_features) == len(self.images), (len(self.clip_features), len(self.images))
        
        ##### Inference input defined here; generate inference images
        ##### Note: why? need further check
        # inference_input = 'apple-red'
        inference_input = 'sports-red'
        inf_data = inference_input.split(",")
        inf_data = [word.replace('_', " ") for word in inf_data]
        inf_data = [word.split('-') for word in inf_data] # [['apple', 'green'], ['apple', 'red'], ...]
        logger.info(f'inf_data: {inf_data}')

        inf_ph_tokens = [[f'mytoken{2*i}', f'mytoken{2*i + 1}'] for i in range(len(inf_data))]
        logger.info(f'inf_ph_tokens: {inf_ph_tokens}')

        assert len(inf_data) == len(inf_ph_tokens), (len(inf_data), len(inf_ph_tokens), inf_data, inf_ph_tokens)

        self.inf_gt_prompts = [self.templates[0].format(*pair) for pair in inf_data]
        logger.info(f'self.inf_gt_prompts: {self.inf_gt_prompts}')

        self.inf_prompts = [self.templates[0].format(*inf_ph_tokens[i]) for i in range(len(inf_data))]
        logger.info(f'self.inf_prompts: {self.inf_prompts}')

        self.inf_fruit_prompts = [imagenet_templates_small[0].format(pair[0]) for pair in inf_ph_tokens]
        self.inf_color_prompts = ['a photo of the color {}'.format(pair[1]) for pair in inf_ph_tokens]
        logger.info(f'self.inf_fruit_prompts: {self.inf_fruit_prompts}')
        logger.info(f'self.inf_color_prompts: {self.inf_color_prompts}')

        self.inf_images, _, _, inf_pil_images = deepfloyd_sample_prompts(self.inf_gt_prompts, num_repeats=1)

        preprocessed_images = self.inf_images.to("cuda")
        preprocessed_images = clip_preprocess(preprocessed_images)

        run_count = 0
        self.inf_clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                self.inf_clip_features.append(result)
                run_count += 1

        self.inf_clip_features = [torch.stack(clip_feature, dim=1).type(torch.float32).to("cpu") for clip_feature in self.inf_clip_features] 
        for i in range(len(self.inf_clip_features)):
            assert self.inf_clip_features[i].shape == (3, 12, 1024)
        assert len(self.inf_clip_features) == len(self.inf_images), (len(self.inf_clip_features), len(self.inf_images))

        self.inf_dict = {
            'image': [img for img in self.inf_images],
            'prompt': self.inf_prompts,
            'gt_prompt': self.inf_gt_prompts,
            'fruit_prompt': self.inf_fruit_prompts,
            'color_prompt': self.inf_color_prompts,
            'clip_feature': [feat for feat in self.inf_clip_features],
        }
        logger.info(f'self.inf_dict: {self.inf_dict}')

        del clip_vision
        torch.cuda.empty_cache()
        logger.info('')
        #########################################################################################################

        ###################### Prepare placeholder words and prompts ######################
        ##### Note: the idea here is to use a fixed size of all possible instances of a concept,
        #####       and then use a placeholder word for each instance.
        #####       e.g, for "color" concept: "red" has a placeholder word, "blue" has another, etc.
        
        self.num_placeholder_words = num_placeholder_words*num_tokens_per_word
        num_placeholder_tokens = num_placeholder_words*num_tokens_per_word*num_placeholder_groups
        placeholder_words = [] #['mytoken0', 'mytoken1', 'mytoken0', ...., ]
        fruit_dict = {}
        fruit_count = 0
        color_count = 1
        for i in range(len(ground_truth_words)):
            curr_fruit = ground_truth_words[i][0]
            if shared_tokens == 1:
                # the category token is shared across different instances of the same category
                ### Note: here is where whether duplicate tokens are used!
                ###       But why not having such option for color as well???
                if curr_fruit not in fruit_dict:
                    fruit_dict[curr_fruit] = f'mytoken{fruit_count}'
                    fruit_count += 2
                placeholder_words.append(fruit_dict[curr_fruit])
            else:
                placeholder_words.append(f'mytoken{fruit_count}')
                fruit_count += 2
            placeholder_words.append(f'mytoken{color_count}')
            color_count += 2
        logger.info(f'placeholder_words: {placeholder_words}')

        placeholder_words = np.split(np.array(placeholder_words), num_placeholder_words*num_placeholder_groups)
        placeholder_words_prompt_args = np.transpose(np.split(np.array(placeholder_words), num_placeholder_words), (1,0,2))
        logger.info(f'placeholder_words: {placeholder_words}')
        logger.info(f'placeholder_words_prompt_args: {placeholder_words_prompt_args}')

        assert len(placeholder_words_prompt_args) == len(ground_truth_prompt_args), (len(placeholder_words_prompt_args), len(ground_truth_prompt_args))
        for placeholder_words_prompt_arg in placeholder_words_prompt_args:
            assert len(placeholder_words_prompt_arg) == num_placeholder_words, (placeholder_words_prompt_arg, num_placeholder_words)
        
        ### Augment the ph_words to align with the generated images (one pair of ph_words per image)
        ### Note that the images are augmented by num_data_per_prompt,
        ###      thus for one unique ph_words pair, there are num_data_per_prompt images, 
        ###      thus needs to augment ph_words by num_data_per_prompt
        self.ph_words_all = []
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            self.ph_words_all.extend([curr_ph_words] * num_data_per_prompt)
        logger.info(f'self.ph_words_all: {self.ph_words_all}')

        self.placeholder_words_prompt_args = placeholder_words_prompt_args
        unique_ph_words = [] # [['mytoken0', 'mytoken1'], ['mytoken2', 'mytoken3']]
        num_placeholder_words = len(self.placeholder_words_prompt_args[0])
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in self.placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            unique_ph_words.append([''.join(word) for word in curr_ph_words])
        logger.info(f'unique_ph_words: {unique_ph_words}')

        blip_fruit_for_each_ph = defaultdict(list)
        assert len(unique_ph_words) == len(self.blip_fruits), (len(unique_ph_words), len(self.blip_fruits), unique_ph_words, self.blip_fruits)
        for i in range(len(ground_truth_words)):
            ph_fruit = unique_ph_words[i][0]
            blip_fruit = self.blip_fruits[i]
            blip_fruit_for_each_ph[ph_fruit].append(blip_fruit)
        logger.info(f'blip_fruit_for_each_ph: {blip_fruit_for_each_ph}')

        common_blip_fruit_for_each_ph = {}
        for ph_fruit in blip_fruit_for_each_ph:
            blip_fruit_counter = Counter(blip_fruit_for_each_ph[ph_fruit])
            blip_common_fruit = blip_fruit_counter.most_common(1)[0][0]
            common_blip_fruit_for_each_ph[ph_fruit] = blip_common_fruit
        logger.info(f'common_blip_fruit_for_each_ph: {common_blip_fruit_for_each_ph}')

        self.blip_fruits = [common_blip_fruit_for_each_ph[ph_pair[0]] for ph_pair in unique_ph_words]
        assert len(unique_ph_words) == len(self.blip_fruits), (len(unique_ph_words), len(self.blip_fruits), unique_ph_words, self.blip_fruits)
        logger.info(f'self.blip_fruits: {self.blip_fruits}')

        self.num_data_copies = num_data_copies
        self.num_data_per_prompt = num_data_per_prompt

        logger.info('Finish checking __init__ for SyntheticBiLevel\n')
        #########################################################################################################

    def __len__(self):
        return len(self.images) * self.num_data_copies

    def __getitem__(self, item):
        item = item % len(self.images)
        # GLIDE expects range [-1, 1]
        image: torch.Tensor = self.images[item]
        if np.random.rand() < .5:
            image = TF.hflip(image)

        curr_ph_words = self.ph_words_all[item]

        template = self.templates[np.random.choice(len(self.templates))]
        prompt = template.format(*[''.join(word) for word in curr_ph_words])

        clip_feature = self.clip_features[item]
        blip_color = self.blip_colors[item//self.num_data_per_prompt]
        blip_fruit = self.blip_fruits[item//self.num_data_per_prompt]

        return {
            'image': image, 
            'prompt': prompt, 
            'gt_prompt': self.gt_prompts[item//self.num_data_per_prompt], 
            'clip_feature': clip_feature, 
            'blip_color': blip_color, 
            'blip_fruit': blip_fruit
        }


class SyntheticBiLevelEval(SyntheticBiLevel):
    def __init__(self, data_root: str, num_placeholder_words: int, templates: Dict, ref_dataset: SyntheticBiLevel):

        unique_ph_words = [] # [['mytoken0', 'mytoken1'], ['mytoken2', 'mytoken3']]
        num_placeholder_words = len(ref_dataset.placeholder_words_prompt_args[0])
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in ref_dataset.placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            unique_ph_words.append([''.join(word) for word in curr_ph_words])

        self.image = torch.zeros_like(ref_dataset.images[0])
        self.gt_word_pairs = ref_dataset.unique_gt_words
        self.ph_word_pairs = unique_ph_words
        self.full_template = ref_dataset.templates[0]
        self.fruit_template = imagenet_templates_small[0]
        self.color_template0 = 'the color {}'
        self.color_template1 = 'a photo of the color {}'
        self.color_template2 = '{}'
        
        self.val_batch_size = 4 ### Here defines how much val smaples per prompt
        self.all_gt_colors = [word_pair[1] for word_pair in self.gt_word_pairs]
        self.all_ph_colors = [word_pair[1] for word_pair in self.ph_word_pairs]
        self.all_colors = [word_pair[1] for word_pair in ref_dataset.unique_gt_words]
        self.blip_colors = ref_dataset.blip_colors
        self.blip_fruits = ref_dataset.blip_fruits


        assert len(self.all_ph_colors) == len(self.all_gt_colors), (len(self.all_ph_colors), len(self.all_gt_colors))

        self.inf_dict = ref_dataset.inf_dict

    def __len__(self):
        return len(self.gt_word_pairs) * self.val_batch_size


    def __getitem__(self, item):
        gt_word_pair = self.gt_word_pairs[item//self.val_batch_size]
        ph_word_pair = self.ph_word_pairs[item//self.val_batch_size]
        gt_prompt = self.full_template.format(*gt_word_pair)
        prompt = self.full_template.format(*ph_word_pair)

        ### Here defines the number of color samples per evaluation - col num of the 
        ### The number is randomly sampled
        random.seed(item)
        assert len(self.all_gt_colors) == len(self.all_ph_colors), (len(self.all_gt_colors), len(self.all_ph_colors), self.all_gt_colors, self.all_ph_colors)
        if item < self.val_batch_size:
            indices = list(range(len(self.all_gt_colors)))
        else:
            indices = sorted(random.sample((list(range(len(self.all_gt_colors)))), 10))
        
        return {
            'image': self.image,
            'prompt': prompt,
            'gt_prompt': gt_prompt,
            'gt_fruit': gt_word_pair[0],
            'gt_color': gt_word_pair[1],
            'ph_fruit': ph_word_pair[0],
            'ph_color': ph_word_pair[1],
            'full_template': self.full_template,
            'fruit_template': self.fruit_template,
            'color_template0': self.color_template0,
            'color_template1': self.color_template1,
            'color_template2': self.color_template2,
            'all_gt_colors': [self.all_gt_colors[i] for i in indices],
            'all_ph_colors': [self.all_ph_colors[i] for i in indices],
            'all_colors': self.all_colors,
            'blip_color': self.blip_colors[item//self.val_batch_size],
            'blip_fruit': self.blip_fruits[item//self.val_batch_size],
            'inf': self.inf_dict
        }
