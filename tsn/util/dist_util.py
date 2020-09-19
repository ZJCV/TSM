# -*- coding: utf-8 -*-

"""
@date: 2020/9/17 下午2:13
@file: dist_util.py
@author: zj
@description: 
"""

import os
import numpy as np
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size, gpus, backend='nccl'):
    # initialize the process group
    dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available() and gpus == 1:
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cleanup():
    dist.destroy_process_group()
