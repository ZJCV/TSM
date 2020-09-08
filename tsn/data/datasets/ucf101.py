# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午10:57
@file: ucf101.py
@author: zj
@description: 
"""

import os

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

    def __init__(self, data_dir, annotation_dir, modality=("RGB"), num_seg=3, splits=(1,), train=True, transform=None):
        assert isinstance(splits, tuple) and len(splits) <= 3
        super(UCF101, self).__init__(data_dir, modality=modality, num_seg=num_seg, transform=transform)

        annotation_list = list()
        if train:
            for split in splits:
                annotation_path = os.path.join(annotation_dir, f'ucf101_train_split_{split}_rawframes.txt')
                if not os.path.isfile(annotation_path):
                    raise ValueError(f'{annotation_path}不是文件路径')

                annotation_list.append(annotation_path)
        else:
            for split in splits:
                annotation_path = os.path.join(annotation_dir, f'ucf101_val_split_{split}_rawframes.txt')
                if not os.path.isfile(annotation_path):
                    raise ValueError(f'{annotation_path}不是文件路径')

                annotation_list.append(annotation_path)

        self.update(annotation_list)
        self.update_class(classes)
