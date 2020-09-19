# -*- coding: utf-8 -*-

"""
@date: 2020/9/10 下午7:43
@file: tsn_recognizer.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry
from tsn.model.backbones.build import build_backbone
from tsn.model.heads.build import build_head
from .temporal_shift import make_temporal_shift


@registry.RECOGNIZER.register('TSNRecognizer')
class TSNRecognizer(nn.Module):

    def __init__(self, cfg, map_location=None):
        super(TSNRecognizer, self).__init__()

        num_segs = cfg.DATASETS.NUM_SEGS

        self.backbone = build_backbone(cfg, map_location=map_location)
        make_temporal_shift(self.backbone, num_segs, place='block')
        self.head = build_head(cfg)

    def forward(self, imgs):
        features = self.backbone(imgs)
        outputs = self.head(features)

        return outputs
