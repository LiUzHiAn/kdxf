import json

import cv2
import glob
import os
from tqdm import tqdm
import jieba

import re


def find_chinese(doc, keep_alphabet=False):
    if not keep_alphabet:
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
    else:
        raise NotImplementedError  # https://www.jb51.net/article/180132.htm
        # pattern = re.compile('r[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]')
    chinese = re.sub(pattern, '', doc)
    return chinese


def find_unchinese(doc):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    unchinese = re.sub(pattern, "", doc)
    return unchinese


def jiaba_seg_docs(train_txt_root="../data/kdxf_cls/training_set_txt",
                   test_txt_root="../data/kdxf_cls/test_set_txt"):
    # # 文本预料库
    # croups = {}
    #
    # # ===================训练集数据==========================
    # for cls_name in os.listdir(train_txt_root):
    #     cls_txt_path = os.path.join(train_txt_root, cls_name)
    #     cls_txts = glob.glob(os.path.join(train_txt_root, cls_name, "*.txt"))
    #     for txt in tqdm(cls_txts, total=len(cls_txts), desc="处理类别%s" % cls_txt_path):
    #         # for txt in cls_txts:
    #         # 读入原始OCR提取的文字
    #         with open(txt, "r") as fp:
    #             doc_raw = fp.read()
    #
    #         # 开始处理doc
    #         doc = "".join(doc_raw.split("||")).replace(' ', '')
    #         # 去掉非中文
    #         doc = find_chinese(doc, keep_alphabet=False)
    #         # print("[原始句子]: ", doc)
    #         seg_list = list(jieba.cut(doc, cut_all=False))  # 精确模式
    #         # print("[默认精确模式]: " + "/ ".join(seg_list))
    #
    #         for token in seg_list:
    #             croups[token] = croups.get(token, 0) + 1
    #
    #         with open(txt.replace(".txt", ".json"), "w") as fp:
    #             json.dump(dict(tokens="||".join(seg_list)), fp, ensure_ascii=False)

    with open("kdxf_croups.json", "r") as fp:
        croups = json.load(fp)

    # ===================测试集数据==========================
    cls_txts = glob.glob(os.path.join(test_txt_root, "*.txt"))
    for txt in tqdm(cls_txts, total=len(cls_txts), desc="处理测试数据"):
        # for txt in cls_txts:
        # 读入原始OCR提取的文字
        with open(txt, "r") as fp:
            doc_raw = fp.read()

        # 开始处理doc
        doc = "".join(doc_raw.split("||")).replace(' ', '')
        # 去掉非中文
        doc = find_chinese(doc, keep_alphabet=False)
        # print("[原始句子]: ", doc)
        seg_list = list(jieba.cut(doc, cut_all=False))  # 精确模式
        # print("[默认精确模式]: " + "/ ".join(seg_list))

        for token in seg_list:
            croups[token] = croups.get(token, 0) + 1

        with open(txt.replace(".txt", ".json"), "w") as fp:
            json.dump(dict(tokens="||".join(seg_list)), fp, ensure_ascii=False)

    # 保存到文件
    with open("kdxf_croups.json", "w") as fp:
        json.dump(croups, fp, ensure_ascii=False)

    return croups


def read_croups():
    # 原始的预料库
    with open("kdxf_croups.json", "r") as fp:
        croups = json.load(fp)

    # 按照出现频率排序
    # croups = sorted(croups.items(), key=lambda x: x[1], reverse=True)

    print(-1)


if __name__ == '__main__':
    jiaba_seg_docs()

    # read_croups()
