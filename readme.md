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