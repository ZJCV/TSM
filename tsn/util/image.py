# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午8:52
@file: image.py
@author: zj
@description: 
"""

import cv2
import numpy as np


def rgbdiff(img1, img2):
    assert len(img1.shape) == 3 and len(img2.shape) == 3

    return np.abs(img1.astype(np.float) - img2.astype(np.float)).astype(np.uint8)
