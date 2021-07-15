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

from mmcls.datasets import DATASETS, BaseDataset


# Regist model so that we can access the class through str in configs
@DATASETS.register_module()
class KDXFDataset(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            # The ann_file is the annotation files we generated. (filename cls_id)
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return [int(self.data_infos[idx]['gt_label'])]


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
        validate=True,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict())
