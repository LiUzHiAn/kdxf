## Data preprocessing


## 0. Dataset

The dataset is collected by the competition offical and I have no right to redistribute to others, so please download it from [here](http://challenge.xfyun.cn/topic/info?type=ad-2021) and place it in the `data`  folder of the root directory in this repo. 

The dataset structure should be similar as follows:
```
./data/kdxf_cls
├── training_set
│   ├── 0
│   │  ├── a.jpg
│   │  ├── b.jpg
│   │  ├── c.jpg
│   ├── 1
│   │  ├── d.jpg
│   │  ├── e.jpg
│   │  ├── f.jpg
│   ├── 2
│   ├── ...
│   ├── 108
├── test_set
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── ...
│   ├── xxx.jpg
```

### 1. OCR

we use [EasyOCR](https://github.com/JaidedAI/EasyOCR) to extract sentences, which is an easy-to-use and supports 80+ languages. 

```python
$ cd pre_process
$ python ocr.py
```

The above script will save the sentences into a .txt file with same name of the image, under the `training_set_txt` and `test_set_txt` folders.

### 2. Tokenization

We use [jibe](https://github.com/fxsjy/jieba) as the tokenizer. whose objective is to segment the sentence into tokens. Suppose you are in the `pre_process` directory, then executing:

```python
$ python jieba_seg.py
```

The above script will make use of the .txt file in OCR phase, resulting in a .json file with the same filename for every single image. Besides, a croups of the dataset will be default pickled into a `kdxf_croups.json` file in current directory.

### 3. Word2Vec

We use [Tencet Word2Vec](https://ai.tencent.com/ailab/nlp/en/embedding.html) model — a large scale pre-trained embedding corpus for both Chinese and English words and phrases — to get the embedding feature vectors. Please download the pre-trained weights in advance (about 8GB). The model is easy-to-use and compatible with [gensim](https://radimrehurek.com/gensim/).

We first extract the word embedding for every token in the dataset croups. To be more concrete:

- For a token exsited in Tencet Word2Vec model, we use the corresponding word vector directly (e.g., 腾讯--> feature vector of 腾讯, Laptop --> feature vector of Laptop);
- For a token non-existed in Tencent Word2Vec model, say 词向量, the feature vector for this kind token is the summurization of each character (i.e., "词向量"="词"+"向"+"量").

```python
$ python tencent_w2v.py
```

The above script will make use of the pickled `kdxf_croups.json` file in tokenization phase, resulting in a `kdxf_w2v_embeddings.pkl` that saves the fixed-length feature vectors of all the tokens in the corups.

> Make sure you transform the original Tencet Word2Vec model into binary format in advance for faster processing. One can run the `load_tencent_w2v()` in `tencent_w2v.py` to fulfill this.

The last thing — extracting the corresponding text feature vector for every single image. To do that, 

```python
$ python extrac_embbeddings.py
```

This script will make use of the `kdxf_w2v_embeddings.pkl` file and the .txt file in OCR phase, resulting in a .npy file for every single image with same name.

### 4. Well-done!

Congrats! You've finished all the preprocessing procedures.

You data structure should be similar as follows:
```
./data/kdxf_cls
├── training_set
│   ├── 0
│   │  ├── a.jpg
│   │  ├── b.jpg
│   │  ├── c.jpg
│   ├── 1
│   │  ├── d.jpg
│   │  ├── e.jpg
│   │  ├── f.jpg
│   ├── 2
│   ├── ...
│   ├── 108
├── training_set_txt
│   ├── 0
│   │  ├── a.txt
│   │  ├── a.json
│   │  ├── a.npy
│   │  ├── b.txt
│   │  ├── b.json
│   │  ├── b.npy
│   ├── 1
│   │  ├── d.txt
│   │  ├── d.json
│   │  ├── d.npy
│   │  ├── e.txt
│   │  ├── e.json
│   │  ├── e.npy
│   ├── 2
│   ├── ...
│   ├── 108
├── test_set
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── ...
│   ├── xxx.jpg
├── test_set_txt
│   ├── 1.txt
│   ├── 1.json
│   ├── 1.npy
│   ├── ...
│   ├── xxx.txt
│   ├── xxx.json
│   ├── xxx.npy
```
