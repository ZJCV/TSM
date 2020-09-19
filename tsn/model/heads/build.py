# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from tsn.model import registry

from .tsn_head import TSNHead


def build_head(cfg):
    return registry.HEAD[cfg.MODEL.HEAD.NAME](cfg)
