<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.en.md">🇺🇸</a>
  <!-- <a title="俄语" href="../ru/README.md">🇷🇺</a> -->
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/TSM"><img align="center" src="./imgs/TSM.png"></a></div>

<p align="center">
  «TSM»复现了论文<a title="" href="https://arxiv.org/abs/1811.08383">Temporal Shift Module</a>提出的视频分类模型
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

其`CodeBase`为[ZJCV/TSN](https://github.com/ZJCV/TSN)

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [使用](#使用)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

[TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383)在`TSN`模型的基础上嵌入时间抖动模块，进一步提高了视频分类精度。其`CodeBase`来自于[TSN](https://github.com/ZJCV/TSN)

## 安装

通过`requirements.txt`安装运行所需依赖

```
$ pip install -r requirements.txt
```

处理数据时需要额外安装[denseflow](https://github.com/open-mmlab/denseflow)，可以在[innerlee/setup](https://github.com/innerlee/setup)中找到安装脚本

## 使用

首先设置`GPU`和当前位置

```
$ export CUDA_VISIBLE_DEVICES=1
$ export PYTHONPATH=.
```

* 训练

```
# 训练UCF101
# 单GPU
$ python tools/train.py --config_file=configs/tsm_resnet50_ucf101_rgb.yaml
# 多GPU
$ python tools/train.py \
--config_file=configs/tsm_resnet50_ucf101_rgb.yaml \
--eval_step=1000 \
--save_step=1000 \
-g=2
```

* 测试

```
# 单模态测试
$ python tools/test.py <config_file> <pth_file>
$ python tools/test.py configs/tsm_resnet50_ucf101_rgbdiff.yaml outputs/tsm_resnet50_ucf101_rgbdiff.pth
# 多模态融合测试 - RGB + RGBDiff
$ python tools/fusion.py <rgb_config_file> <rgb_pth_file> <rgbdiff_config_file> <rgbdiff_pth_file>
$ python tools/fusion.py \
configs/tsm_resnet50_ucf101_rgb.yaml \
outputs/tsm_resnet50_ucf101_rgb.pth  \
configs/tsm_resnet50_ucf101_rgbdiff.yaml \
outputs/tsm_resnet50_ucf101_rgbdiff.pth
```


## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

* [mit-han-lab/temporal-shift-module ](https://github.com/mit-han-lab/temporal-shift-module)
* [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjykzj/TSM/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjykzj