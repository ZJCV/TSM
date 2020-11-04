# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午4:45
@file: mobilenet.py
@author: zj
@description: 
"""

import torchvision.models as models
from torchvision.models.utils import load_state_dict_from_url

from .. import registry

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def mobilenet_v2(pretrained=False, progress=True, map_location=None, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = models.MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress,
                                              map_location=map_location)
        model.load_state_dict(state_dict)
    return model


@registry.BACKBONE.register('MobileNet_v2')
def build_mobilenet_v2(cfg, map_location=None):
    pretrained = cfg.MODEL.BACKBONE.TORCHVISION_PRETRAINED

    model = mobilenet_v2(pretrained=pretrained, map_location=map_location)
    return model.features
