# 提取每张图片对应的doc的词向量,策略是将所有的词向量求平均作为该doc的词向量
import os
import json
import glob

import numpy as np
from tqdm import tqdm
import cv2
import pickle
import matplotlib.pyplot as plt


def get_training_set_embeddings(vis_emb=False):
    with open("kdxf_w2v_embeddings.pkl", "rb") as fp:
        tokens_info = pickle.load(fp)

    train_txt_root = "../data/kdxf_cls/training_set_txt"
    for cls_name in os.listdir(train_txt_root):
        cls_txt_path = os.path.join(train_txt_root, cls_name)

        docs = glob.glob(os.path.join(train_txt_root, cls_name, "*.json"))
        for doc_path in tqdm(docs, total=len(docs)):
            emb_save_file = os.path.join(cls_txt_path, os.path.basename(doc_path).replace(".json", ""))

            with open(doc_path, "r") as fp:
                doc_tokens = json.load(fp)["tokens"].split("||")

            # 没识别到词,用全0表示
            if len(doc_tokens) == 1 and doc_tokens[0] == '':
                doc_emb = np.zeros(200)
            else:
                doc_emb = np.zeros(200)
                num_token_valid = 0
                for token in doc_tokens:
                    # 第1类token
                    if len(tokens_info[token]["idx"]) == 1:
                        doc_emb += tokens_info[token]["word_emb"][0]
                        num_token_valid += 1
                doc_emb = doc_emb / num_token_valid if num_token_valid != 0 else np.zeros(200)

            np.save(emb_save_file, doc_emb)

            # 可视化embedding
            if vis_emb:
                fig = plt.figure()
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.tile(doc_emb, 10).reshape(10, -1))  # 复制10行,便于可视化
                plt.colorbar()

                # redraw the canvas
                fig.canvas.draw()
                # convert canvas to image
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                plt.close(fig)
                cv2.imshow("emb", img)
                cv2.waitKey(0)


def get_test_set_embeddings(vis_emb=False):
    with open("kdxf_w2v_embeddings.pkl", "rb") as fp:
        tokens_info = pickle.load(fp)

    test_txt_root = "../data/kdxf_cls/test_set_txt"

    docs = glob.glob(os.path.join(test_txt_root, "*.json"))
    for doc_path in tqdm(docs, total=len(docs)):
        emb_save_file = doc_path.replace(".json", "")

        with open(doc_path, "r") as fp:
            doc_tokens = json.load(fp)["tokens"].split("||")

        # 没识别到词,用全0表示
        if len(doc_tokens) == 1 and doc_tokens[0] == '':
            doc_emb = np.zeros(200)
        else:
            doc_emb = np.zeros(200)
            num_token_valid = 0
            for token in doc_tokens:
                # 第1类token
                if len(tokens_info[token]["idx"]) == 1:
                    doc_emb += tokens_info[token]["word_emb"][0]
                    num_token_valid += 1
            doc_emb = doc_emb / num_token_valid if num_token_valid != 0 else np.zeros(200)

        np.save(emb_save_file, doc_emb)

        # 可视化embedding
        if vis_emb:
            fig = plt.figure()
            plt.xticks([])
            plt.yticks([])
            plt.imshow(np.tile(doc_emb, 10).reshape(10, -1))  # 复制10行,便于可视化
            plt.colorbar()

            # redraw the canvas
            fig.canvas.draw()
            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            plt.close(fig)
            cv2.imshow("emb", img)
            cv2.waitKey(0)


if __name__ == '__main__':
    get_training_set_embeddings(vis_emb=False)
    get_test_set_embeddings(vis_emb=False)

    arr = np.load("/home/liuzhian/hdd4T/code/kdxf/data/kdxf_cls/training_set_txt/113/12539.npy")
    assert arr.shape[0] == 200
    print(-1)
