# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午6:47
@file: consensus.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class Consensus(nn.Module):

    def __init__(self, type='avg', dim=0):
        super(Consensus, self).__init__()
        self.type = type
        self.dim = dim

    def forward(self, input):
        assert isinstance(input, torch.Tensor)

        self.shape = input.size()
        if self.type == 'avg':
            output = input.mean(dim=self.dim, keepdim=False)
            return output
        else:
            raise ValueError('融合类型不存在')
