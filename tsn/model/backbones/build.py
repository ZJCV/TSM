# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

from tsn.util.distributed import get_device, get_local_rank

from .. import registry
from .mobilenet import build_mobilenet_v2
from .shufflenet import build_shufflenet_v2
from .resnet import build_resnet50


def build_backbone(cfg):
    device = get_device(local_rank=get_local_rank())
    return registry.BACKBONE[cfg.MODEL.BACKBONE.NAME](cfg, map_location=device)
