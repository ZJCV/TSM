# -*- coding: utf-8 -*-

"""
@date: 2020/9/20 下午4:23
@file: tsm_recognizer.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry
from tsn.model.backbones.build import build_backbone
from tsn.model.heads.build import build_head
from .temporal_shift import make_temporal_shift


@registry.RECOGNIZER.register('TSMRecognizer')
class TSMRecognizer(nn.Module):

    def __init__(self, cfg, map_location=None):
        super(TSMRecognizer, self).__init__()

        num_segs = cfg.DATASETS.NUM_SEGS

        self.backbone = build_backbone(cfg, map_location=map_location)
        make_temporal_shift(self.backbone, num_segs, place='block')
        self.head = build_head(cfg)

    def forward(self, imgs):
        features = self.backbone(imgs)
        outputs = self.head(features)

        return outputs
