# 1. 提取图片文字
先用OCR提取图片中的文字,`ocr.py`中的`get_test_set_ocr()`和`get_train_set_ocr()`,
每张图片保存一个对应的txt文件,用`||`分割识别到的每一处文字

# 2. 用jieba分词
用jieba分词,先去除原始数据中非中文字符,再分词,每个原始的txt文件对应生成一个.json文件.
见`jieba_seg.json`中的`jiaba_seg_docs()`,分词的同时,会统计所有数据集的预料库,
得到一个dict,key为token,val 

# 3. 得到预料库的词向量
基于上述预料库,对于每个单词,去腾讯的w2v模型中找对应的词向量.
- 如果某个词存在,则直接用该词向量;
- 如果不存在,则把该词组中的每个字对应的词向量相加作为词向量.(例如`"词向量"="词"+"向"+"量"`)

# 4. 提取每张图片对应doc的词向量
对于一个doc中的所有词, 会存在两种词:
1. 这个词在腾讯w2v中直接存在;
2. 这个词在腾讯w2v中不存在;

现在的方案是:
- 对于第2类词,直接丢掉,只将第1类词求平均,保存到一个`.npy`文件中

详情见`extrac_embbeddings.py`中的`get_training_set_embeddings()`和`get_test_set_embeddings()`函数 