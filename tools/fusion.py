# -*- coding: utf-8 -*-

"""
@date: 2020/9/18 上午9:53
@file: fusion.py
@author: zj
@description: 
"""

import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from datetime import datetime

from tsn.config import get_cfg_defaults
from tsn.data.build import build_dataloader
from tsn.model.build import build_model
from tsn.util.metrics import topk_accuracy
from tsn.util.checkpoint import CheckPointer
from tsn.util.logger import setup_logger
from tsn.util.collect_env import collect_env_info


@torch.no_grad()
def compute_on_dataset(rgb_model, rgb_data_loader, rgbdiff_model, rgbdiff_data_loader, device):
    results_dict = {}
    cate_acc_dict = {}
    acc_top1 = list()
    acc_top5 = list()

    cpu_device = torch.device("cpu")
    rgb_data_loader_iter = iter(rgb_data_loader)
    rgbdiff_data_loader_iter = iter(rgbdiff_data_loader)
    for i in tqdm(range(len(rgb_data_loader))):
        outputs_list = list()

        images, targets = next(rgb_data_loader_iter)
        outputs = rgb_model(images.to(device)).to(cpu_device)
        outputs_list.append(outputs)

        images, targets = next(rgbdiff_data_loader_iter)
        outputs = rgbdiff_model(images.to(device)).to(cpu_device)
        outputs_list.append(outputs)
        outputs = torch.mean(torch.stack(outputs_list), dim=0)

        topk_list = topk_accuracy(outputs, targets, topk=(1, 5))
        acc_top1.append(topk_list[0].item())
        acc_top5.append(topk_list[1].item())

        outputs = outputs.numpy()
        preds = np.argmax(outputs, 1)
        targets = targets.numpy()
        for target, pred in zip(targets, preds):
            results_dict.update({
                str(target):
                    results_dict.get(str(target), 0) + 1
            })
            cate_acc_dict.update({
                str(target):
                    cate_acc_dict.get(str(target), 0) + int(target == pred)
            })

    return results_dict, cate_acc_dict, acc_top1, acc_top5


def inference(rgb_cfg, rgb_model, rgbdiff_cfg, rgbdiff_model, device):
    logger_name = rgb_cfg.INFER.NAME
    dataset_name = rgb_cfg.DATASETS.TEST.NAME
    output_dir = rgb_cfg.OUTPUT.DIR

    rgb_data_loader = build_dataloader(rgb_cfg, train=False)
    rgbdiff_data_loader = build_dataloader(rgbdiff_cfg, train=False)
    dataset = rgb_data_loader.dataset

    logger = setup_logger(logger_name)
    logger.info("Evaluating {} dataset({} video clips):".format(dataset_name, len(dataset)))

    results_dict, cate_acc_dict, acc_top1, acc_top5 = \
        compute_on_dataset(rgb_model, rgb_data_loader, rgbdiff_model, rgbdiff_data_loader, device)

    top1_acc = np.mean(acc_top1)
    top5_acc = np.mean(acc_top5)
    result_str = '\ntotal - top_1 acc: {:.3f}, top_5 acc: {:.3f}\n'.format(top1_acc, top5_acc)

    classes = dataset.classes
    for key in sorted(results_dict.keys(), key=lambda x: int(x)):
        total_num = results_dict[key]
        acc_num = cate_acc_dict[key]

        cate_name = classes[int(key)]

        if total_num != 0:
            result_str += '{:<3} - {:<20} - acc: {:.2f}\n'.format(key, cate_name, acc_num / total_num * 100)
        else:
            result_str += '{:<3} - {:<20} - acc: 0.0\n'.format(key, cate_name, acc_num / total_num)
    logger.info(result_str)

    result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    return {'top1': top1_acc, 'top5': top5_acc}


def test(args):
    torch.backends.cudnn.benchmark = True
    logger = setup_logger('TEST')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}

    # # 计算RGB
    rgb_cfg = get_cfg_defaults()
    rgb_cfg.merge_from_file(args.rgb_config_file)
    rgb_cfg.DATALOADER.TEST_BATCH_SIZE = 16
    rgb_cfg.OUTPUT.DIR = args.output
    rgb_cfg.freeze()

    rgb_model = build_model(rgb_cfg, map_location=map_location).to(device)
    rgb_model.eval()
    checkpointer = CheckPointer(rgb_model, logger=logger)
    checkpointer.load(args.rgb_pretrained, map_location=map_location)

    # inference(rgb_cfg, rgb_model, device)

    # 计算RGBDiff
    rgbdiff_cfg = get_cfg_defaults()
    rgbdiff_cfg.merge_from_file(args.rgbdiff_config_file)
    rgbdiff_cfg.DATALOADER.TEST_BATCH_SIZE = 16
    rgbdiff_cfg.OUTPUT.DIR = args.output
    rgbdiff_cfg.freeze()

    rgbdiff_model = build_model(rgbdiff_cfg, map_location=map_location).to(device)
    rgbdiff_model.eval()
    checkpointer = CheckPointer(rgbdiff_model, logger=logger)
    checkpointer.load(args.rgbdiff_pretrained, map_location=map_location)

    inference(rgb_cfg, rgb_model, rgbdiff_cfg, rgbdiff_model, device)


def main():
    parser = argparse.ArgumentParser(description='TSN Test With PyTorch')
    parser.add_argument("rgb_config_file", default="", metavar="RGB_CONFIG_FILE",
                        help="path to config file", type=str)
    parser.add_argument('rgb_pretrained', default="", metavar='RGB_PRETRAINED_FILE',
                        help="path to pretrained model", type=str)
    parser.add_argument("rgbdiff_config_file", default="", metavar="RGBDIFF_CONFIG_FILE",
                        help="path to config file", type=str)
    parser.add_argument('rgbdiff_pretrained', default="", metavar='RGBDIFF_PRETRAINED_FILE',
                        help="path to pretrained model", type=str)
    parser.add_argument('--output', default="./outputs/test", type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.rgb_config_file) and not os.path.isfile(args.rgb_pretrained):
        raise ValueError('需要输入RGB模态配置文件和预训练模型路径')
    if not os.path.isfile(args.rgbdiff_config_file) or not os.path.isfile(args.rgbdiff_pretrained):
        raise ValueError('需要输入RGBDIFF模态配置文件和预训练模型路径')

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    logger = setup_logger("TSN", save_dir=args.output)
    logger.info(args)
    logger.info("Environment info:\n" + collect_env_info())

    test(args)


if __name__ == '__main__':
    main()
