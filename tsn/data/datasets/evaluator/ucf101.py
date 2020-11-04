# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 上午10:09
@file: ucf101.py
@author: zj
@description: 
"""

import torch

from .base_evaluator import BaseEvaluator
from tsn.util.metrics import topk_accuracy


class UCF101Evaluator(BaseEvaluator):

    def __init__(self, classes, topk=(1,)):
        super().__init__(classes)

        self.topk = topk
        self._init()

    def _init(self):
        self.topk_list = list()
        self.cate_acc_dict = dict()
        self.cate_num_dict = dict()

    def evaluate_train(self, output_dict: dict, targets: torch.Tensor):
        assert isinstance(output_dict, dict) and 'probs' in output_dict.keys()

        probs = output_dict['probs']
        res = topk_accuracy(probs, targets, topk=self.topk)

        acc_dict = dict()
        for i in range(len(self.topk)):
            acc_dict[f'tok{self.topk[i]}'] = res[i]
        return acc_dict

    def evaluate_test(self, output_dict: dict, targets: torch.Tensor):
        assert isinstance(output_dict, dict) and 'probs' in output_dict.keys()
        probs = output_dict['probs']
        outputs = probs.to(device=self.device)
        targets = targets.to(device=self.device)

        res = topk_accuracy(outputs, targets, topk=self.topk)
        self.topk_list.append(torch.stack(res))
        preds = torch.argmax(outputs, dim=1)
        for target, pred in zip(targets.numpy(), preds.numpy()):
            self.cate_num_dict.update({
                str(target):
                    self.cate_num_dict.get(str(target), 0) + 1
            })
            self.cate_acc_dict.update({
                str(target):
                    self.cate_acc_dict.get(str(target), 0) + int(target == pred)
            })

    def get(self):
        if len(self.topk_list) == 0:
            return None, None

        cate_topk_dict = dict()
        for key in self.cate_num_dict.keys():
            total_num = self.cate_num_dict[key]
            acc_num = self.cate_acc_dict[key]
            class_name = self.classes[int(key)]

            cate_topk_dict[class_name] = 1.0 * acc_num / total_num if total_num != 0 else 0.0

        result_str = '\ntotal -'
        acc_dict = dict()
        topk_list = torch.mean(torch.stack(self.topk_list), dim=0)
        for i in range(len(self.topk)):
            acc_dict[f"top{self.topk[i]}"] = topk_list[i]
            result_str += ' {} acc: {:.3f}'.format(f"top{self.topk[i]}", topk_list[i])
        result_str += '\n'

        for idx in range(len(self.classes)):
            class_name = self.classes[idx]
            cate_acc = cate_topk_dict[class_name]

            if cate_acc != 0:
                result_str += '{:<3} - {:<20} - acc: {:.2f}\n'.format(idx, class_name, cate_acc * 100)
            else:
                result_str += '{:<3} - {:<20} - acc: 0.0\n'.format(idx, class_name)

        return result_str, acc_dict

    def clean(self):
        self._init()
