# *BERT-Clinical-NER*

### *Introduction*

基于BERT预训练语言模型的中文电子病历命名实体识别　

### *数据集和模型结构*

本项目使用的数据集来自CCKS2019医疗命名实体识别子任务一；模型中的BERT预训练语言模型可以更好地表示电子病历句子中的上下文语义,迭代膨胀卷积神经网络(IDCNN)对局部实体的卷积编码有更好的识别效果,多头注意力机制(MHA)多次计算每个字和所有字的注意力概率以获取电子病历句子的长距离依赖。

### *运行步骤*

1. 下载[CCKS2019医疗命名实体识别数据集](https://www.biendata.xyz/competition/ccks_2019_1/)，放入data目录；
2. 下载[BERT中文预训练语言模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)，解压后放入根目录；
3. 训练模型:python3 train.py
4. 模型预测：python3 predict.py

### *Requirements*

1. python3
2. tensorflow-gpu >= 1.10

### *Reference*

(1) https://github.com/zjy-ucas/ChineseNER

(2) https://github.com/google-research/bert

(3) https://github.com/RacleRay/NER_project

(4) https://github.com/beyondguo/JD_CV_Match