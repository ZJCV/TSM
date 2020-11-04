# -*- coding: utf-8 -*-

"""
@date: 2020/9/10 下午7:43
@file: tsn_recognizer.py
@author: zj
@description: 
"""

import torch.nn as nn

from .. import registry
from ..backbones.build import build_backbone
from ..heads.build import build_head
from ..consensus.build import build_consensus


@registry.RECOGNIZER.register('TSNRecognizer')
class TSNRecognizer(nn.Module):

    def __init__(self, cfg):
        super(TSNRecognizer, self).__init__()

        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)
        self.consensus = build_consensus(cfg)

    def forward(self, imgs):
        assert len(imgs.shape) == 5
        imgs = imgs.transpose(1, 2)
        N, T, C, H, W = imgs.shape[:5]

        input_data = imgs.reshape(-1, C, H, W)
        features = self.backbone(input_data)
        probs = self.head(features).reshape(N, T, -1)

        probs = self.consensus(probs, dim=1)
        return {'probs': probs}
