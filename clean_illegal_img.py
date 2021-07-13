# 数据集里有空的文件
import cv2
import glob
import os

train_root = "./data/kdxf_cls/training_set"
for cls_path in os.listdir(train_root):
    cls_imgs = glob.glob(os.path.join(train_root, cls_path, "*.jpg"))
    for img_path in cls_imgs:
        img = cv2.imread(img_path)
        if img is None:
            print(img_path)

print(-1)