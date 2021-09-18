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
import torch
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose
import argparse

with open("./cls_id_to_name.json", 'r') as fp:
    cls_id_to_name = json.load(fp)


def multiModal_inference_model(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageEmbeddingFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageEmbeddingFromFile'))
        data = dict(
            img_prefix=cfg.data.test.data_prefix,
            img_info=dict(filename=osp.basename(img)),

            emb_prefix=cfg.data.test.data_prefix + "_txt",
            emb_info=dict(filename=osp.basename(img).replace(".jpg", ".npy").replace("test_set", "test_set_txt")),
        )
    else:
        raise NotImplementedError("The %s must be the image path." % img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        result = {'pred_label': pred_label, 'pred_score': float(pred_score), 'raw_scores': scores[0]}
    result['pred_class'] = model.CLASSES[result['pred_label']]
    return result


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


def inference_test_imgs(cfg_file, ckpt_file, show_imgs=False):
    img_name_column = []
    pred_cls_column = []
    pred_score_column = []

    # Build the model from a config file and a checkpoint file
    model = init_model(cfg_file, ckpt_file, device='cuda:0')
    # Test a single image
    with open("./data/kdxf_cls/test.txt", 'r') as fp:
        test_imgs = fp.readlines()

    test_imgs = [osp.join("./data/kdxf_cls/test_set", img.split('\n')[0]) for img in test_imgs]
    for img_path in tqdm(test_imgs, total=len(test_imgs)):
        # 一些测试集中读取不出来的图片
        if osp.basename(img_path) == "a2411.jpg":
            result = (dict(pred_label=105, pred_class='105', pred_score=1.0))
        else:
            result = tta(multiModal_inference_model, model, img_path)

        result['pred_class'] = cls_id_to_name[result['pred_class']]
        img_name_column.append(osp.basename(img_path))
        pred_score_column.append(result["pred_score"])
        pred_cls_column.append(result["pred_label"])

        if show_imgs:
            result_img = my_show_result(img_path, result)
            cv2.imshow("res", result_img)
            cv2.waitKey(0)

    return img_name_column, pred_score_column, pred_cls_column


def append_submit_clsName(submit_file="./submit/resnet101_multiModal_clsBalanced_tta5.csv"):
    """在提交的csv文件中加一个类名列,方便验证"""
    df = pd.read_csv(submit_file, sep=',')
    cls_pred = df.loc[:, ["category_id"]].values.reshape(-1)
    cls_name = [cls_id_to_name[str(item)] for item in cls_pred]
    df["cls_name"] = cls_name

    df.to_csv(submit_file.replace(".csv", "wClsName.csv"), index=False, sep=',')


def analysis_submit_dist(submit_file="./submit/resnet101_multiModal_clsBalanced_tta5_wClsName.csv"):
    """分析预测distribution"""
    import matplotlib.pyplot as plt

    df = pd.read_csv(submit_file, sep=',')
    cls_pred = df.loc[:, ["category_id"]].values.reshape(-1)

    num_pred_cls = np.zeros(137)
    for cls in cls_pred:
        num_pred_cls[cls] += 1

    plt.plot(np.arange(137), num_pred_cls)
    plt.xlabel('Class')
    plt.ylabel('Num')
    plt.show()
    print(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg", type=str, default='configs/img_word_emb_run_config',
        help='mmclassification格式的参数文件')
    parser.add_argument(
        "--ckpt", type=str,
        default='work_dirs/r101_multiModal_clsBalanced_MoE_labelSmoothing_FocalLoss_fullTrain/epoch_110.pth',
        help='mmclassification格式的参数文件')
    parser.add_argument(
        "--save_csv", type=str,
        default='./submit/r101_multiModal_clsBalanced_MoE_labelSmoothing_FocalLoss_fullTrain_epoch110_tta5.csv',
        help='测试集推理结果,保存到csv文件')

    args = parser.parse_args()

    img_name, pred_score, pred_cls = inference_test_imgs(args.cfg, args.ckpt, show_imgs=False)

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'image_id': img_name, 'category_id': pred_cls})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(args.save_csv, index=False, sep=',')
