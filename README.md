# KDXF-image-classification Competition

This repo contains the Top-25% solution of [广告图片素材分类算法挑战赛](http://challenge.xfyun.cn/topic/info?type=ad-2021), 
our team name is 对对对对对.
![](assets/demo-fig.jpg)
## 1. Introduction
As a basic task in computer vision, the task of image classification is to 
predict the corresponding category when given an input image. We consider that the text is 
helpful modality that would help to classify better, just as what the human-beings exactly think.
To this end, we train a classification model that leverages the image and text modalities together.

- For the image modality, we adopt the CNN (ResNet in actual) as the image feature extractor, feel free to
  inject the recent Transformer-based model to get better performance.
- For the text modality we adopt the `word2vec` model to extract the underlying text features in an image. Specifically, given an 
input image, we first use some OCR tools to extract the sentences in an image. Then, we use a tokenizer to segment the 
sentences into tokens. Finally, each single token is projected to the embedding vector by a word2vec model. The projected
token features are aggregated, obtaining the text features as another input source for the classification model.

## 2. Prerequisite
Our implementation is based on [MMClassification](https://github.com/open-mmlab/mmclassification), 
please checkout my kdxf_dev branch from [here](https://github.com/LiUzHiAn/mmclassification/tree/kdxf_dev).

The detailed requirements are:
```
matplotlib==3.3.4
numpy==1.19.2
jieba==0.42.1
annoy==1.17.0
mmcls.egg==info
torch==1.7.0
mmcv_full==1.3.9
tqdm==4.61.2
gensim==4.0.1
opencv_python==4.5.3.56
pandas==1.1.5
easyocr==1.3.2
mmcls==0.15.0
mmcv==1.3.13
Pillow==8.3.2
```

## 3. Data preprocess
Please follow the instructions [here](./pre_process/README.md) to prepare the KDXF dataset and finish the preprocessing procedure.

There are totally 137 categories in the KDXF dataset. It is worthy noting that the actual category names 
are not provided, just simply annotated as 0,1,2,...,136 instead. I manually check some images and provide the class names in `data/kdxf_cls/cls_id_to_name.json`,
some selected classes are:
```
"0": "化妆品",
"1": "淘宝/京东电商",
"2": "饿了么外卖",
"3": "母婴奶粉",
"4": "手机",
"5": "抖音快手",
"6": "口红",
"7": "母婴纸尿裤",
"8": "女装(衣服裤子等)",
"9": "小型厨房电器(微波炉,豆浆机等)",
"10": "家电(电风扇,扫地机器人等)",
...
```
## 4. Train & Test
### 4.1 Dataset splitting
We use 15% of the original training set as our validation set, which is use for model selecting. Just run:
```python
$ python split_dataset.py [--split <split_frac>]
```
The default split fraction is set to be 0.85. This script will generate the corresponding image names of train/dev/test dataset
in `data/kdxf_cls/train.txt`,`data/kdxf_cls/val.txt` and `data/kdxf_cls/test.txt`, respectively.

> In the final stage, we use both the train and validation set to train a model with more data, 
> which is used to submit our competition results.  

### 4.2 Training
```python
$ python train.py [--cfg <path/to/config>] [--validate 0]
```
The default config file is set as `img_word_emb_run_config.py`, which strictly follows the config 
style of MMClassification, feel free to modify that accordingly.

The above scripts will train the model with 110 epochs and save the best model using the validation set.
If you want to use both of the train and validation set to train the model, just specify the `validate` to 0 
and replace the `ann_file='./data/kdxf_cls/train.txt'` with  `ann_file='./data/kdxf_cls/train_full.txt'` 
in Line37 of `configs/datasets/img_word_emb_dataset_config.py`
 

Some highlights:
- We extract the image features using `ResNet101`, which is pretrained on ImageNet;
- We extract the text features using `Word2Vec`;
- Since the number of images are uneven among different categories, we balance the dataset using the `ClassBalancedDataset`
wrapper provided by MMClassification;
- The image and text modalities are fused using a `Mixture-of-Experts(MoE)` classifier;
- We use the `cross-entropy loss` to train the model, and further add `label smooth` and `focal loss` tricks to obtain better results;
- During inference, we adopt the `Test-Time-Augmentation(TTA)` trick to boost the performance. To do this, just turn on the
Line20-Line21 of `configs/datasets/img_word_emb_dataset_config.py`;

### 4.3 Testing
```python
$ python inference.py --cfg <path/to/config> \
                    --ckpt <path/to/ckpt> \
                    --save_csv <path/to/csv>
```
This script will save the inference result in a .csv file you specify, according to the given model config and ckpt weights.
## 5. Result
We get 87.221% Top-1 classification accuracy on the test set currently.