import pandas as pd
from mmcls.apis import inference_model, init_model, show_result_pyplot
import json
import mmcv
from mmcv import color_val
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os.path as osp
from tqdm import tqdm

with open("./data/kdxf_cls/cls_id_to_name.json", 'r') as fp:
    cls_id_to_name = json.load(fp)


def my_show_result(img, result):
    """可视化预测结果,支持中文标签"""
    img = mmcv.imread(img)
    img = img.copy()

    # write results on left-top of the image
    x, y = 0, 20
    text_color = color_val('green')

    fontStyle = ImageFont.truetype("./work_dirs/simsun.ttc", 20, encoding="utf-8")
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for k, v in result.items():
        if isinstance(v, float):
            v = f'{v:.2f}'
        label_text = f'{k}: {v}'

        draw.text((x, y), label_text, text_color, font=fontStyle)
        y += 20

    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


def tta(func, model, img_path, num_aug=5):
    """测试时增强,测试num_aug次,求平均"""
    probs = []
    for i in range(num_aug):
        tmp_result = func(model, img_path)
        tmp_probs = tmp_result["raw_scores"]
        probs.append(tmp_probs)

    probs = np.mean(np.array(probs), axis=0)
    result = dict()
    result["pred_score"] = np.max(probs)
    result["pred_label"] = np.argmax(probs)
    result["pred_class"] = str(result["pred_label"])
    return result


def inference_test_imgs(show_imgs=False):
    img_name_column = []
    pred_cls_column = []
    pred_score_column = []

    # Specify the path to config file and checkpoint file
    config_file = './configs/resnet101_b32x8_imagenet.py'
    checkpoint_file = 'work_dirs/resnet101_clsBalanced/latest.pth'

    # Build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    # Test a single image
    with open("./data/kdxf_cls/test.txt", 'r') as fp:
        test_imgs = fp.readlines()

    test_imgs = [osp.join("./data/kdxf_cls/test_set", img.split('\n')[0]) for img in test_imgs]
    for img_path in tqdm(test_imgs, total=len(test_imgs)):
        # 一些测试集中读取不出来的图片
        if osp.basename(img_path) == "a2411.jpg":
            result = (dict(pred_label=105, pred_class='105', pred_score=1.0))
        else:
            result = tta(inference_model, model, img_path)

        result['pred_class'] = cls_id_to_name[result['pred_class']]
        img_name_column.append(osp.basename(img_path))
        pred_score_column.append(result["pred_score"])
        pred_cls_column.append(result["pred_label"])

        if show_imgs:
            result_img = my_show_result(img_path, result)
            cv2.imshow("res", result_img)
            cv2.waitKey(0)

    return img_name_column, pred_score_column, pred_cls_column


if __name__ == '__main__':
    img_name, pred_score, pred_cls = inference_test_imgs(show_imgs=False)
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'image_id': img_name, 'category_id': pred_cls})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("./submit/resnet101_clsBalanced_tta5.csv", index=False, sep=',')
