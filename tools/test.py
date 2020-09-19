# -*- coding: utf-8 -*-

"""
@date: 2020/9/18 上午9:15
@file: test.py
@author: zj
@description: 
"""

import os
import numpy as np
import torch
import argparse

from tsn.config import cfg
from tsn.data.build import build_dataloader
from tsn.model.build import build_model
from tsn.engine.trainer import do_train
from tsn.engine.inference import do_evaluation
from tsn.util.checkpoint import CheckPointer
from tsn.util.logger import setup_logger
from tsn.util.collect_env import collect_env_info


def test(cfg):
    torch.backends.cudnn.benchmark = True

    logger = setup_logger('TEST')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}

    model = build_model(cfg, map_location=map_location).to(device)
    if cfg.MODEL.PRETRAINED != "":
        if logger:
            logger.info(f'load pretrained: {cfg.MODEL.PRETRAINED}')
        checkpointer = CheckPointer(model, logger=logger)
        checkpointer.load(cfg.MODEL.PRETRAINED, map_location=map_location)

    do_evaluation(cfg, model, device)


def main():
    parser = argparse.ArgumentParser(description='TSN Test With PyTorch')
    parser.add_argument("config_file", default="", metavar="CONFIG_FILE",
                        help="path to config file", type=str)
    parser.add_argument('pretrained', default="", metavar='PRETRAINED_FILE',
                        help="path to pretrained model", type=str)
    parser.add_argument('--output', default="./outputs/test", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config_file) or not os.path.isfile(args.pretrained):
        raise ValueError('需要输入配置文件和预训练模型路径')

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.PRETRAINED = args.pretrained
    cfg.OUTPUT.DIR = args.output
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

    test(cfg)


if __name__ == '__main__':
    main()
