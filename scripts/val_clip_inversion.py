import sys
sys.path.insert(0, '/users/ljunyu/data/ljunyu/code/concept/deepfloyd/')

import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
from tu.utils.config import build_from_config
from tu.train.setup import get_cfg, get_parser
from tu.train_setup import open_tensorboard, build_dataloaders, build_training_modules, setup_ddp_safe
from tu.utils.config import overwrite_cfg
from pathlib import Path
from tu.trainers.simple_trainer import load_checkpoint

import logging
logger = logging.getLogger(__name__)

def main():
    rank = setup_ddp_safe()
    print(rank)

    parser = get_parser()
    args = parser.parse_args()

    # Setup configuration
    cfg = get_cfg(args)

    # If running in a specific environment like debugging or special hardware setup
    if os.getenv('DEBUG') == '1':
        torch.autograd.set_detect_anomaly(True)
        overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'print_every', 10)  # Adjust if needed

    # Load data loaders
    _, val_loader = build_dataloaders(cfg, seed=args.seed, use_ddp=False)  # Assuming no distributed computing for validation

    # Build models and load checkpoint
    modules, optimizers, schedulers = build_training_modules(cfg, use_ddp=False)
    
    log_dir = cfg['log_dir']
    if rank == 0:
        writer = SummaryWriter(log_dir)
        open_tensorboard(log_dir)
        logger.info(f'tensorboard --bind_all --logdir {Path(log_dir).absolute()}')
    else:
        writer = None

    trainer = build_from_config(cfg['trainer'], modules=modules, writer=writer, optimizers=optimizers, schedulers=schedulers)

    # checkpoint_path = cfg['training']['checkpoint_dir']
    checkpoint_path = '/users/ljunyu/data/ljunyu/code/concept/langint-train-two/logs/__train_deepfloyd_inversion_car-two-train-debug-pipeline2/checkpoints/it_00004000.pt'
    print(checkpoint_path)
    load_checkpoint(trainer, cfg, checkpoint_path)

    # Run validation
    trainer.validate(val_loader)

if __name__ == "__main__":
    main()