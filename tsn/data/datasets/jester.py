# -*- coding: utf-8 -*-

"""
@date: 2020/9/18 下午4:18
@file: jester.py
@author: zj
@description: 
"""

import os
import cv2
import torch
from PIL import Image
import numpy as np
from .base_dataset import VideoRecord
from .base_dataset import BaseDataset


class JESTER(BaseDataset):

    def __init__(self, data_dir, annotation_dir,
                 modality="RGB", num_segs=3, train=True, transform=None):
        super(JESTER, self).__init__(data_dir, train=train, modality=modality, num_segs=num_segs, transform=transform)

        label_path = os.path.join(annotation_dir, 'jester-v1-labels.csv')
        classes = list(np.loadtxt(label_path, dtype=np.str, delimiter=','))
        self._update_class(classes)

        if train:
            anno_path = os.path.join(annotation_dir, 'jester-v1-train.csv')
        else:
            anno_path = os.path.join(annotation_dir, 'jester-v1-validation.csv')

        video_list = list()
        anno_array = np.loadtxt(anno_path, dtype=np.str, delimiter=';')
        for anno in anno_array:
            path = anno[0]
            label_name = anno[1]
            label = classes.index(label_name)

            data_path = os.path.join(data_dir, path)
            num_frames = len(os.listdir(data_path))

            video_list.append(VideoRecord([path, num_frames, label]))
        self.video_list = video_list

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
            if num == 0:
                num += 1
            if 'RGB' == self.modality:
                image_path = os.path.join(video_path, '{:0>5d}.jpg'.format(num))
                img = cv2.imread(image_path)

                if self.transform:
                    img = self.transform(img)
                image_list.append(img)
            if 'RGBDiff' == self.modality:
                tmp_list = list()
                for clip in range(self.clip_length):
                    img_path = os.path.join(video_path, '{:0>5d}.jpg'.format(num + clip))
                    img = np.array(Image.open(img_path))

                    tmp_list.append(img)
                for clip in reversed(range(1, self.clip_length)):
                    img = tmp_list[clip] - tmp_list[clip - 1]
                    if self.transform:
                        img = self.transform(img)
                    image_list.append(img)
        image = torch.stack(image_list)

        return image, target
