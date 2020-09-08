# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午5:06
@file: hmdb51.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms

from tsn.data.datasets.hmdb51 import HMDB51


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return transform


if __name__ == '__main__':
    transform = get_transform()

    data_set = HMDB51('/home/zj/zhonglian/mmaction2/data/hmdb51/rawframes', '/home/zj/zhonglian/mmaction2/data/hmdb51',
                      split=1, num_seg=8, train=True, transform=transform)
    print(data_set)
    print(len(data_set))
    image, target = data_set.__getitem__(100)
    print(image.shape)
    print(target)
