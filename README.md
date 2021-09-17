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
Please follow the instructions [here](./pre_process/README.md) to prepare the dataset and finish the preprocessing procedure.

## 4. Train & Test

## 5. Performance
We use top-1 accuracy to evaluate the model.