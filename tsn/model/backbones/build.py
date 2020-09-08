# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn
from .resnet import resnet50


def build_backbone(name, num_classes=1000, pretrained=True, partial_bn=False):
    if 'resnet50'.__eq__(name):
        model = resnet50(pretrained=pretrained, partial_bn=partial_bn)

        fc = model.fc
        in_features = fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        return model
    else:
        raise ValueError('no matching backbone exists')
