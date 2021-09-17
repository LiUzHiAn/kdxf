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
1. 在多模态的head加上label smoothing; 
2. MoE, mixture of experts;已经写好了.

1和2一起跑, 大概率是label smoothing的作用

3. Focal Loss
==> 用 BalancedDataset做数据增强, 用Focal loss学习难样本,写了一个`LabelSmoothFocalLoss`类  ==> 感觉没啥用
   
4. 把验证集的数据也丢进来训练
==> 有用, 但是没办法选最好的模型,只能选最后一轮
   
5. 换backbone,试下densenet,Xception等,甚至将两个网络的特征融合
==> 换了resNet152, 全量数据集训练,但是效果不太好, 估计是lr没搞对
   
6. bs设置成64
==> 结果差很多,没办法控制
   
7. BalancedDataset Repeat-factor乘以2倍,100轮
=> 没用
   
8. 人工看了一下前1500张图片,用的是resnet101,多模态,MoE, label smooth, classBalanced, epoch110的结果
==> 新建了一个`r101_multiModal_clsBalanced_MoE_labelSmoothing_FocalLoss_fullTrain_epo110_tta5_ML.csv`,
   ML代表,Manually labelled, 详细修改情况见 `r101_multiModal_clsBalanced_MoE_labelSmoothing_FocalLoss_fullTrain_epo110_tta5wClsName.ods`文件.
   标了1500张, 精度反而下降了!!! 
   标3000张试试.


   
