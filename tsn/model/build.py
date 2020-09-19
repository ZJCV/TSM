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


def build_model(cfg, map_location=None):
    model = TSN(cfg, map_location=map_location)
    return model


def build_criterion(cfg):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg)
