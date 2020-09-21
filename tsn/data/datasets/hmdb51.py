# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午4:37
@file: hmdb51.py
@author: zj
@description: 
"""

import os
import cv2
import torch
from PIL import Image
import numpy as np
from .base_dataset import BaseDataset

classes = ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb',
           'climb_stairs', 'dive', 'draw_sword', 'dribble', 'drink', 'eat',
           'fall_floor', 'fencing', 'flic_flac', 'golf', 'handstand', 'hit',
           'hug', 'jump', 'kick', 'kick_ball', 'kiss', 'laugh', 'pick',
           'pour', 'pullup', 'punch', 'push', 'pushup', 'ride_bike',
           'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow',
           'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault',
           'stand', 'swing_baseball', 'sword', 'sword_exercise', 'talk',
           'throw', 'turn', 'walk', 'wave']


class HMDB51(BaseDataset):

    def __init__(self, data_dir, annotation_dir, modality="RGB", num_segs=3, split=1, train=True, transform=None):
        assert isinstance(split, int) and split in (1, 2, 3)
        super(HMDB51, self).__init__(data_dir, train=train, modality=modality, num_segs=num_segs, transform=transform)

        if train:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_train_split_{split}_rawframes.txt')
            if not os.path.isfile(annotation_path):
                raise ValueError(f'{annotation_path}不是文件路径')
        else:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_val_split_{split}_rawframes.txt')
            if not os.path.isfile(annotation_path):
                raise ValueError(f'{annotation_path}不是文件路径')
        self._update(annotation_path)
        self._update_class(classes)

    def __getitem__(self, index: int):
        """
        从选定的视频文件夹中随机选取T帧，则返回(T, C, H, W)，其中T表示num_segs
        """
        assert index < len(self.video_list)
        record = self.video_list[index]
        target = record.label

        if self.train:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        video_path = os.path.join(self.data_dir, record.path)
        image_list = list()
        for num in segment_indices:
            if 'RGB' == self.modality:
                image_path = os.path.join(video_path, 'img_{:0>5d}.jpg'.format(num))
                img = cv2.imread(image_path)

                if self.transform:
                    img = self.transform(img)
                image_list.append(img)
            if 'RGBDiff' == self.modality:
                tmp_list = list()
                for clip in range(self.clip_length):
                    img_path = os.path.join(video_path, 'img_{:0>5d}.jpg'.format(num + clip))
                    img = np.array(Image.open(img_path))

                    tmp_list.append(img)
                for clip in reversed(range(1, self.clip_length)):
                    img = tmp_list[clip] - tmp_list[clip - 1]
                    if self.transform:
                        img = self.transform(img)
                    image_list.append(img)
        image = torch.stack(image_list)

        return image, target