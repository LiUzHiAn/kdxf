# 怎么用(大致)
1. 从官网安装`mmcv`
2. 从[这里](https://github.com/LiUzHiAn/mmclassification) 从源码安装`mmclassification`到本地,并checkout到`kdxf_dev`分支
3. 训练脚本见`train.py`或`dummy.py`.

认真hack一下mmclassification的官方教程,上手很快.

## 纯分类方案 
resnet101暴力分类

## OCR文字识别
用的是[easyOCR](https://github.com/JaidedAI/EasyOCR)

## 待考虑的方向
1. 在多模态的head加上label smoothing; TODO
2. MoE, mixture of experts;已经写好了.TODO

1和2一起跑

3. 把验证集的数据也丢进来训练
4. 换backbone,试下densenet,Xception等,甚至将两个网络的特征融合