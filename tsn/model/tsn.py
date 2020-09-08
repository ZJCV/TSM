# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午9:54
@file: tsn.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from .backbones.build import build_backbone
from .consensus import Consensus


class TSN(nn.Module):

    def __init__(self, num_classes=1000, backbone='resnet50', consensus='avg', partial_bn=True, pretrained=True):
        super(TSN, self).__init__()

        self.backbone = build_backbone(backbone,
                                            num_classes=num_classes, pretrained=pretrained, partial_bn=partial_bn)
        self.consensus = Consensus(type=consensus)

    def forward(self, x):
        """
        输入数据大小为NxTxCxHxW，按T维度分别计算NxCxHxW，然后按照融合策略计算最终分类概率
        上述这一步的实现逻辑有点绕：
        1. 按照上述的实现结果，经过Backbone计算后得到的是TxNxNum_classes，然后按第0维进行融合得到NxNum_classes
        2. 其中可以按N维度计算TxCXHxW，得到NxTxNum_classes，然后按照第1维进行融合得到NxNum_classes
        两者效果是一样的
        """
        assert len(x.shape) == 5
        N, T, C, H, W = x.shape[:5]

        input_data = x.transpose(0, 1)
        prob_list = list()
        for data in input_data:
            prob_list.append(self.backbone(data))

        probs = self.consensus(torch.stack(prob_list))
        return probs