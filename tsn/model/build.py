# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: build.py
@author: zj
@description: 
"""

from . import registry
from .tsn import TSN
from .criterions.crossentropy import build_crossentropy


def build_model(cfg):
    num_classes = cfg.MODEL.NUM_CLASSES
    backbone = cfg.MODEL.BACKBONE
    consensus = cfg.MODEL.CONSENSUS
    partial_bn = cfg.MODEL.PARTIAL_BN
    pretrained = cfg.MODEL.PRETRAINED

    return TSN(num_classes=num_classes, backbone=backbone,
               consensus=consensus, partial_bn=partial_bn, pretrained=pretrained)


def build_criterion(cfg):
    return registry.CRITERIONS[cfg.CRITERION.NAME](cfg)
