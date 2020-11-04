# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

from .. import registry

from .avg_consensus import AvgConsensus


def build_consensus(cfg):
    return registry.CONSENSU[cfg.MODEL.CONSENSU.NAME](cfg)
