# -*- coding: utf-8 -*-

"""
@date: 2020/11/4 下午2:06
@file: compute_flops.py
@author: zj
@description: 
"""

import time
import torch

from tsn.util.distributed import get_device, get_local_rank
from tsn.util.metrics import compute_num_flops
from tsn.config import cfg
from tsn.model.recognizers.build import build_recognizer


def main(data_shape, config_file, mobile_name):
    cfg.merge_from_file(config_file)

    device = get_device(local_rank=get_local_rank())
    model = build_recognizer(cfg, device)
    data = torch.randn(data_shape).to(device=device, non_blocking=True)

    GFlops, params_size = compute_num_flops(model, data)
    print(f'{mobile_name} ' + '*' * 10)
    print(f'device: {device}')
    print(f'GFlops: {GFlops}')
    print(f'Params Size: {params_size}')

    total_time = 0.0
    num = 100
    for i in range(num):
        data = torch.randn(data_shape).to(device=device, non_blocking=True)
        start = time.time()
        model(data)
        total_time = time.time() - start
    print(f'one process need {total_time / num}')


if __name__ == '__main__':
    data_shape = (1, 3, 4, 256, 256)

    sf_cfg = 'configs/tsn_sfv2_ucf101_rgb_raw_dense_1x16x4.yaml'
    sf_name = 'ShuffleNet_v2'
    main(data_shape, sf_cfg, sf_name)

    mb_cfg = 'configs/tsn_mbv2_ucf101_rgb_raw_dense_1x16x4.yaml'
    mb_name = 'MobileNet_v2'
    main(data_shape, mb_cfg, mb_name)

    r50_cfg = 'configs/tsn_r50_ucf101_rgb_raw_dense_1x16x4.yaml'
    r50_name = 'ResNet50'
    main(data_shape, r50_cfg, r50_name)
