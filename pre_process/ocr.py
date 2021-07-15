import easyocr
# 数据集里有空的文件
import cv2
import glob
import os
from tqdm import tqdm


def get_training_set_ocr():
    reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory

    train_root = "../data/kdxf_cls/training_set"
    txt_root = "../data/kdxf_cls/training_set_txt"
    os.makedirs(txt_root, exist_ok=True)

    for cls_name in os.listdir(train_root):
        cls_txt_path = os.path.join(txt_root, cls_name)
        os.makedirs(cls_txt_path, exist_ok=True)
        cls_imgs = glob.glob(os.path.join(train_root, cls_name, "*.jpg"))
        for img_path in tqdm(cls_imgs, total=len(cls_imgs)):
            txts_save_file = os.path.join(cls_txt_path, os.path.basename(img_path).replace(".jpg", ".txt"))

            img = cv2.imread(img_path)
            result = reader.readtext(img)

            if len(result) != 0:
                txts = list(list(zip(*result))[1])
            else:
                txts = []

            with open(txts_save_file, "w") as fp:
                fp.write('||'.join(txts))

            # cv2.imshow("test", cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
            # cv2.waitKey(0)


def get_test_set_ocr():
    reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory

    root = "../data/kdxf_cls/test_set"
    txt_root = "../data/kdxf_cls/test_set_txt"
    os.makedirs(txt_root, exist_ok=True)

    imgs = glob.glob(os.path.join(root, "*.jpg"))
    for img_path in tqdm(imgs, total=len(imgs)):

        txts_save_file = os.path.join(txt_root, os.path.basename(img_path).replace(".jpg", ".txt"))
        # 测试集中的 a2411.jpg读不出来,强制为空
        if os.path.basename(img_path) == "a2411.jpg":
            result = []
        else:
            img = cv2.imread(img_path)
            result = reader.readtext(img)

        if len(result) != 0:
            txts = list(list(zip(*result))[1])
        else:
            txts = []

        with open(txts_save_file, "w") as fp:
            fp.write('||'.join(txts))

        # cv2.imshow("test", cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
        # cv2.waitKey(0)


if __name__ == '__main__':
    get_test_set_ocr()
