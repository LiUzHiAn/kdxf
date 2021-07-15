# 提取每张图片对应的doc的词向量,策略是将所有的词向量求平均作为该doc的词向量
import os
import json
import glob

import numpy as np
from tqdm import tqdm
import cv2
import pickle
import matplotlib.pyplot as plt


def dataset_stats():
    stats = {}
    num_total = 0
    train_imgs_root = "../data/kdxf_cls/training_set"
    for cls_name in os.listdir(train_imgs_root):
        cls_txt_path = glob.glob(os.path.join(train_imgs_root, cls_name, "*.jpg"))
        stats[cls_name] = len(cls_txt_path)
        num_total += len(cls_txt_path)

    for key in stats.keys():
        stats[key] /= num_total

    return stats


if __name__ == '__main__':
    stats = dataset_stats()

    print(-1)
