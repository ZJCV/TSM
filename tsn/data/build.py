# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: build.py
@author: zj
@description: 
"""

import torch
from torch.utils.data import DataLoader

from .datasets.build import build_dataset
from .samplers import IterationBasedBatchSampler

from tsn.data.datasets.hmdb51 import HMDB51
from tsn.data.datasets.ucf101 import UCF101

from .transforms.build import build_transform


def build_dataloader(cfg, train=True):
    transform = build_transform(cfg, train=train)
    dataset = build_dataset(cfg, transform=transform, is_train=train)

    if train:
        # 训练阶段使用随机采样器
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        batch_size = cfg.DATALOADER.TEST_BATCH_SIZE

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    if train:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.TRAIN.MAX_ITER, start_iter=0)

    data_loader = DataLoader(dataset, num_workers=cfg.DATALOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                             pin_memory=True)

    return data_loader
