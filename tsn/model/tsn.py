# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午9:54
@file: tsn.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from .recognizers.build import build_recognizer
from .consensus.build import build_consensus


class TSN(nn.Module):

    def __init__(self, cfg, map_location=None):
        super(TSN, self).__init__()

        self.recognizer = build_recognizer(cfg, map_location=map_location)
        self.consensus = build_consensus(cfg)

    def forward(self, imgs):
        assert len(imgs.shape) == 5
        N, T, C, H, W = imgs.shape[:5]

        input_data = imgs.reshape(-1, C, H, W)
        probs = self.recognizer(input_data).reshape(N, T, -1)

        return self.consensus(probs, dim=1)
