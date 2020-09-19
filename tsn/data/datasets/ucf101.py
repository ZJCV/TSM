# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午10:57
@file: ucf101.py
@author: zj
@description: 
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from .base_dataset import BaseDataset

classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling',
           'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball',
           'BasketballDunk', 'BenchPress', 'Biking', 'Billiards',
           'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling',
           'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke',
           'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling',
           'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing',
           'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch',
           'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow',
           'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
           'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
           'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope',
           'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade',
           'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars',
           'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol',
           'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
           'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse',
           'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor',
           'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
           'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling',
           'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
           'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus',
           'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking',
           'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']


class UCF101(BaseDataset):

    def __init__(self, data_dir, annotation_dir, modality="RGB", num_segs=3, split=1, train=True, transform=None):
        assert isinstance(split, int) and split in (1, 2, 3)
        super(UCF101, self).__init__(data_dir, train=train, modality=modality, num_segs=num_segs, transform=transform)

        if train:
            annotation_path = os.path.join(annotation_dir, f'ucf101_train_split_{split}_rawframes.txt')
            if not os.path.isfile(annotation_path):
                raise ValueError(f'{annotation_path}不是文件路径')
        else:
            annotation_path = os.path.join(annotation_dir, f'ucf101_val_split_{split}_rawframes.txt')
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
