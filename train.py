import time
import os.path as osp
import argparse
from mmcv import Config
from mmcls.apis import set_random_seed
from mmcls.datasets import build_dataset
from mmcls.apis import train_model

import mmcv
import torch
from mmcls.models.classifiers import WordEmbeddingClassifier
from mmcls.models import build_classifier


def model_test():
    """模型测试"""
    cfg = Config.fromfile('configs/models/img_word_emb_model_config.py')

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    print(f'Config:\n{cfg.pretty_text}')

    model = build_classifier(cfg.model)

    out = model(torch.ones(64, 3, 224, 224), torch.ones(64, 200), gt_label=torch.ones(64).long())
    print(out)


def dataset_test():
    """dataset测试"""
    cfg = Config.fromfile('configs/datasets/img_word_emb_dataset_config.py')
    print(f'Config:\n{cfg.pretty_text}')

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    print(-1)


def img_word_emb_run_test(cfg_file, validate):
    cfg = Config.fromfile(cfg_file)

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for finetuning
    print(f'Config:\n{cfg.pretty_text}')

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 保存运行配置文件
    with open(osp.join(cfg.work_dir, "exp_conf.py"), "w") as fp:
        fp.write(cfg.pretty_text)

    # Build the classifier
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
        validate=validate,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default='configs/img_word_emb_run_config',
                        help='mmclassification格式的参数文件')
    parser.add_argument('--validate', type=int, default=0, help='是否使用验证集')
    args = parser.parse_args()

    img_word_emb_run_test(args.cfg, args.validate == 1)
