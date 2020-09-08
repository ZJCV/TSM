# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:26
@file: resnet.py
@author: zj
@description: 
"""

import torch.nn as nn
from tsn.model.backbones.build import build_backbone
from tsn.model.backbones.resnet import BasicBlock, Bottleneck, ResNet

if __name__ == '__main__':
    model = build_backbone('resnet50')
    # print(model)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            print('bn')
        elif isinstance(m, Bottleneck):
            print('bottleneck')
        else:
            pass
