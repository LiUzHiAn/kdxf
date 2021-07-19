import time
import os.path as osp

import mmcv
from mmcv import Config
from mmcls.apis import set_random_seed
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.apis import train_model

import mmcv
import numpy as np

if __name__ == '__main__':
    cfg = Config.fromfile('configs/resnet101_b32x8_imagenet.py')

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for finetuning
    print(f'Config:\n{cfg.pretty_text}')

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # Build the classifiers
    model = build_classifier(cfg.model)
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Begin finetuning
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict())
