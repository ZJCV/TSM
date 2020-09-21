# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: build.py
@author: zj
@description: 
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets.build import build_dataset
from .samplers import IterationBasedBatchSampler

from tsn.data.datasets.hmdb51 import HMDB51
from tsn.data.datasets.ucf101 import UCF101

from .transforms.build import build_transform


def build_dataloader(cfg, train=True,
                     start_iter=0,
                     world_size=1, rank=0):
    transform = build_transform(cfg, train=train)
    dataset = build_dataset(cfg, transform=transform, is_train=train)

    if train:
        if world_size != 1 and rank == 0:
            batch_size = int(cfg.DATALOADER.TRAIN_BATCH_SIZE * 3 / 4)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE
            # 训练阶段使用随机采样器
            sampler = torch.utils.data.RandomSampler(dataset)
    else:
        if world_size != 1 and rank == 0:
            batch_size = int(cfg.DATALOADER.TEST_BATCH_SIZE * 3 / 4)
        else:
            batch_size = cfg.DATALOADER.TEST_BATCH_SIZE
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    if train:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.TRAIN.MAX_ITER,
                                                   start_iter=start_iter)

    data_loader = DataLoader(dataset, num_workers=cfg.DATALOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                             pin_memory=True)

    return data_loader
