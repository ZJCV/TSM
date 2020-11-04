# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午4:47
@file: shufflenet.py
@author: zj
@description: 
"""

import torchvision.models as models
from torchvision.models.utils import load_state_dict_from_url

from .. import registry

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


class ShuffleNetV2(models.ShuffleNetV2):

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        # x = x.mean([2, 3])  # globalpool
        # x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _shufflenetv2(arch, pretrained, progress, map_location, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress, map_location=map_location)
            model.load_state_dict(state_dict)

    return model


def shufflenet_v2_x1_0(pretrained=False, progress=True, map_location=None, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress, map_location,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


@registry.BACKBONE.register('ShuffleNet_v2')
def build_shufflenet_v2(cfg, map_location=None):
    pretrained = cfg.MODEL.BACKBONE.TORCHVISION_PRETRAINED

    return shufflenet_v2_x1_0(pretrained=pretrained, map_location=map_location)
