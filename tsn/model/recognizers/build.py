# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from tsn.model import registry

from .tsn_recognizer import TSNRecognizer
from .tsm_recognizer import TSMRecognizer


def build_recognizer(cfg, map_location=None):
    return registry.RECOGNIZER[cfg.MODEL.RECOGNIZER.NAME](cfg, map_location=map_location)
