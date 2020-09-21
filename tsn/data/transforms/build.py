# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms


def build_transform(cfg, train=True):
    size = cfg.TRANSFORM.INPUT_SIZE
    h, w, c = size
    smaller_edge = cfg.TRANSFORM.SMALLER_EDGE

    MEAN = cfg.TRANSFORM.MEAN
    STD = cfg.TRANSFORM.STD

    if train:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(smaller_edge),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    return transform
