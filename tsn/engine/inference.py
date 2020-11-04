# -*- coding: utf-8 -*-

"""
@date: 2020/8/23 上午9:51
@file: inference.py
@author: zj
@description: 
"""

import os
import datetime
import torch
from tqdm import tqdm

import tsn.util.logging as logging
from tsn.util.distributed import all_gather, is_master_proc
from tsn.data.build import build_dataloader


@torch.no_grad()
def compute_on_dataset(images, targets, device, model, num_gpus, evaluator):
    images = images.to(device=device, non_blocking=True)
    targets = targets.to(device=device, non_blocking=True)

    output_dict = model(images)
    # Gather all the predictions across all the devices to perform ensemble.
    if num_gpus > 1:
        keys = list()
        values = list()
        for key in sorted(output_dict):
            keys.append(key)
            values.append(output_dict[key])
        values = all_gather(values)
        output_dict = {k: v for k, v in zip(keys, values)}
        targets = all_gather([targets])[0]

    evaluator.evaluate_test(output_dict, targets)


def inference(cfg, model, device, **kwargs):
    iteration = kwargs.get('iteration', None)
    dataset_name = cfg.DATASETS.TEST.NAME
    num_gpus = cfg.NUM_GPUS

    data_loader = build_dataloader(cfg, is_train=False)
    dataset = data_loader.dataset
    evaluator = data_loader.dataset.evaluator
    evaluator.clean()

    logger = logging.setup_logging(__name__)
    logger.info("Evaluating {} dataset({} video clips):".format(dataset_name, len(dataset)))

    if is_master_proc():
        for images, targets in tqdm(data_loader):
            compute_on_dataset(images, targets, device, model, num_gpus, evaluator)
    else:
        for images, targets in data_loader:
            compute_on_dataset(images, targets, device, model, num_gpus, evaluator)

    result_str, acc_dict = evaluator.get()
    logger.info(result_str)

    if is_master_proc():
        output_dir = cfg.OUTPUT_DIR
        result_path = os.path.join(output_dir,
                                   'result_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))) \
            if iteration is None else os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))

        with open(result_path, "w") as f:
            f.write(result_str)

    return acc_dict


@torch.no_grad()
def do_evaluation(cfg, model, device, **kwargs):
    model.eval()

    return inference(cfg, model, device, **kwargs)
