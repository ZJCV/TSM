# -*- coding: utf-8 -*-

"""
@date: 2020/9/18 上午9:15
@file: test.py
@author: zj
@description: 
"""

import numpy as np
import torch

from tsn.model.recognizers.build import build_recognizer
from tsn.engine.inference import do_evaluation
from tsn.util.collect_env import collect_env_info
from tsn.util import logging
from tsn.util.distributed import get_device, get_local_rank
from tsn.util.parser import parse_test_args, load_test_config
from tsn.util.misc import launch_job
from tsn.util.distributed import synchronize, init_distributed_training


def test(cfg):
    # Set up environment.
    init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = get_device(local_rank=get_local_rank())
    model = build_recognizer(cfg, device=device)

    synchronize()
    do_evaluation(cfg, model, device)


def main():
    args = parse_test_args()
    cfg = load_test_config(args)

    logger = logging.setup_logging(__name__, output_dir=cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == '__main__':
    main()
