# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:21
@file: trainer.py
@author: zj
@description: 
"""

from .hmdb51 import HMDB51
from .ucf101 import UCF101
from .jester import JESTER


def build_dataset(cfg, transform=None, is_train=True):
    dataset_name = cfg.DATASETS.TRAIN.NAME if is_train else cfg.DATASETS.TEST.NAME
    data_dir = cfg.DATASETS.TRAIN.DATA_DIR if is_train else cfg.DATASETS.TEST.DATA_DIR
    annotation_dir = cfg.DATASETS.TRAIN.ANNOTATION_DIR if is_train else cfg.DATASETS.TEST.ANNOTATION_DIR

    modality = cfg.DATASETS.MODALITY
    num_segs = cfg.DATASETS.NUM_SEGS

    if dataset_name == 'HMDB51':
        split = cfg.DATASETS.TRAIN.SPLIT if is_train else cfg.DATASETS.TEST.SPLIT

        dataset = HMDB51(data_dir, annotation_dir, train=is_train, modality=modality, num_segs=num_segs,
                         split=split, transform=transform)
    elif dataset_name == 'UCF101':
        split = cfg.DATASETS.TRAIN.SPLIT if is_train else cfg.DATASETS.TEST.SPLIT

        dataset = UCF101(data_dir, annotation_dir, train=is_train, modality=modality, num_segs=num_segs,
                         split=split, transform=transform)
    elif dataset_name == 'JESTER':
        dataset = JESTER(data_dir, annotation_dir, train=is_train, modality=modality, num_segs=num_segs,
                         transform=transform)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset
