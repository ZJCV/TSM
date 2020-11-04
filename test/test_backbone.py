# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午5:07
@file: test_backbone.py
@author: zj
@description: 
"""

import torch
from tsn.config import cfg

from tsn.model.backbones.build import build_resnet50, build_mobilenet_v2, build_shufflenet_v2
from tsn.model.backbones.regnet import build_regnet


def test_resnet50():
    cfg.merge_from_file('configs/tsn_r50_ucf101_rgb_raw_dense_1x16x4.yaml')

    model = build_resnet50(cfg)
    data = torch.randn((1, 3, 224, 224))
    outputs = model(data)

    print(outputs.shape)
    assert outputs.shape == (1, 2048, 7, 7)


def test_mobilenet():
    cfg.merge_from_file('configs/tsn_mbv2_ucf101_rgb_raw_dense_1x16x4.yaml')

    model = build_mobilenet_v2(cfg)
    data = torch.randn((1, 3, 224, 224))
    outputs = model(data)

    print(outputs.shape)
    assert outputs.shape == (1, 1280, 7, 7)


def test_shufflenet():
    cfg.merge_from_file('configs/tsn_sfv2_ucf101_rgb_raw_dense_1x16x4.yaml')

    model = build_shufflenet_v2(cfg)
    data = torch.randn((1, 3, 224, 224))
    outputs = model(data)

    print(outputs.shape)
    assert outputs.shape == (1, 1024, 7, 7)


if __name__ == '__main__':
    test_resnet50()
    test_mobilenet()
    test_shufflenet()
