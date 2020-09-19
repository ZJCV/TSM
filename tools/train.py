# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: build.py
@author: zj
@description: 
"""

import os
import numpy as np
import torch
import argparse

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tsn.config import cfg
from tsn.data.build import build_dataloader
from tsn.model.build import build_model, build_criterion
from tsn.optim.build import build_optimizer, build_lr_scheduler
from tsn.engine.trainer import do_train
from tsn.engine.inference import do_evaluation
from tsn.util.checkpoint import CheckPointer
from tsn.util.logger import setup_logger
from tsn.util.collect_env import collect_env_info
from tsn.util.dist_util import setup, cleanup


def train(gpu, args, cfg):
    rank = args.nr * args.gpus + gpu
    setup(rank, args.world_size, args.gpus)

    logger = setup_logger(cfg.TRAIN.NAME)
    arguments = {"iteration": 0}
    arguments['rank'] = rank

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    model = build_model(cfg, map_location=map_location).to(device)
    if cfg.MODEL.PRETRAINED != "":
        if rank == 0 and logger:
            logger.info(f'load pretrained: {cfg.MODEL.PRETRAINED}')
        checkpointer = CheckPointer(model, logger=logger)
        checkpointer.load(cfg.MODEL.PRETRAINED, map_location=map_location, rank=rank)

    if args.gpus > 1:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=cfg.OUTPUT.DIR,
                                save_to_disk=True, logger=logger)
    if args.resume:
        if rank == 0:
            logger.info('resume ...')
        extra_checkpoint_data = checkpointer.load(map_location=map_location, rank=rank)
        if extra_checkpoint_data != dict():
            arguments['iteration'] = extra_checkpoint_data['iteration']
            if cfg.LR_SCHEDULER.WARMUP:
                if rank == 0:
                    logger.info('warmup ...')
                if lr_scheduler.finished:
                    optimizer.load_state_dict(lr_scheduler.after_scheduler.optimizer.state_dict())
                else:
                    optimizer.load_state_dict(lr_scheduler.optimizer.state_dict())
                lr_scheduler.optimizer = optimizer
                lr_scheduler.after_scheduler.optimizer = optimizer

    data_loader = build_dataloader(cfg, train=True, start_iter=arguments['iteration'],
                                   gpus=args.gpus, world_size=args.world_size, rank=rank)

    model = do_train(args, cfg, arguments,
                     data_loader, model, criterion, optimizer, lr_scheduler,
                     checkpointer, device, logger)

    if rank == 0 and not args.stop_eval:
        logger.info('Start final evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, device)

    cleanup()


def main():
    parser = argparse.ArgumentParser(description='TSN Training With PyTorch')
    parser.add_argument("--config_file", default="", metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument('--log_step', default=10, type=int,
                        help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int,
                        help='Save checkpoint every save_step')
    parser.add_argument('--stop_save', default=False, action='store_true')
    parser.add_argument('--eval_step', default=2500, type=int,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--stop_eval', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume training')
    parser.add_argument('--use_tensorboard', default=1, type=int)

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of machines (default: 1)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT.DIR):
        os.makedirs(cfg.OUTPUT.DIR)
    logger = setup_logger("TSN", save_dir=cfg.OUTPUT.DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '17928'
    mp.spawn(train, nprocs=args.gpus, args=(args, cfg))


if __name__ == '__main__':
    main()
