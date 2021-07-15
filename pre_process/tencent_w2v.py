import pickle

import numpy as np
from gensim.models.word2vec import KeyedVectors
from collections import OrderedDict
import json
from annoy import AnnoyIndex

PRE_TRAINED_FILE = "/home/liuzhian/hdd4T/pretrained_models/tencent_cn_embedding/Tencent_AILab_ChineseEmbedding.bin"
EMB_DIM = 200


def load_tencent_w2v():
    # tc_wv_model = KeyedVectors.load_word2vec_format(PRE_TRAINED_FILE, binary=False, limit=4000000)
    tc_wv_model = KeyedVectors.load(PRE_TRAINED_FILE)
    # 保存为二进制
    # tc_wv_model.save(PRE_TRAINED_FILE.replace(".txt", ".bin"))

    with open("tc_w2v_key2idx.json", "w") as fp:
        json.dump(tc_wv_model.key_to_index, fp)


def get_needed_tokens_idx():
    """得到kdxf训练和测试集中所有单词,在腾讯w2v模型中的索引"""

    with open("tc_w2v_key2idx.json", "r") as fp:
        tc_wv_key_to_index = json.load(fp)

    # kdxf数据集的原始预料库(包括训练集和测试集)
    with open("kdxf_croups.json", "r") as fp:
        croups = json.load(fp)

    croups = sorted(croups.items(), key=lambda x: x[1], reverse=True)

    tokens_info = {}

    for word, cnt in croups:
        # 如果该词存在于腾讯的词向量中,就保存一下该词对应的id
        if tc_wv_key_to_index.get(word) is not None:
            idx = [tc_wv_key_to_index[word]]

        else:
            # TODO: 得到每个字对应的索引,如果不存在,用-1表示
            idx = [tc_wv_key_to_index[word[i]] if tc_wv_key_to_index.get(word[i]) is not None else -1
                   for i in range(len(word))]

        tokens_info[word] = dict(idx=idx)

    return tokens_info


def get_tokens_embeddings():
    """得到kdxf训练和测试集中所有单词,在腾讯w2v模型中的 词向量"""
    tokens_info = get_needed_tokens_idx()
    tc_wv_model = KeyedVectors.load(PRE_TRAINED_FILE)

    for key in tokens_info.keys():
        # 腾讯w2v中存在该词,好办,直接找出embedding feats
        if len(tokens_info[key]["idx"]) == 1:
            token_emb = tc_wv_model.vectors[tokens_info[key]["idx"]]
        else:
            token_emb = np.zeros(200)
            for idx in tokens_info[key]["idx"]:  # 把几个字的词向量加起来
                token_emb += tc_wv_model.vectors[idx]

        tokens_info[key]["word_emb"] = token_emb

    with open("kdxf_w2v_embeddings.pkl", "wb") as fp:
        pickle.dump(tokens_info, fp)


if __name__ == '__main__':
    # get_tokens_embeddings()

    with open("kdxf_w2v_embeddings.pkl", "rb") as fp:
        tokens_info = pickle.load(fp)
    print(-1)