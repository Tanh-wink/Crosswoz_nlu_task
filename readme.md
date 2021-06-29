### 1、Crosswoz NLU

利用中文大规模对话数据集Crosswoz，进行NLU任务实验。

### 2、数据集Crosswoz

数据集Crosswoz是2020年清华大学人工智能研究院发布的一个大规模中文对话数据集，可应用于任务型对话系统中各个部分，包括NLU、DST、对话策略学习、NLG等。
论文：[CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset](https://arxiv.org/abs/2002.11893)

### 3、NLU任务说明

数据集：Crosswoz数据
    -训练集：5012个对话，84692个utterance（句子）
    -验证集：500个对话，8458个utterance（句子）
    -测试集：500个对话，8400个utterance（句子）
槽位提取模型：bert + Bilstm + crf
intent 模型：

### 4、环境需求

python >= 3.5

pytorch==1.9.0

numpy

transformers==3.0.0

tqdm==4.59.0

### 5、使用

（1）新建data目录，下载数据集放在该目录下：https://github.com/thu-coai/CrossWOZ

（2）首先运行 preprocess/preprocess.py 对下载回来的crosswoz进行预处理为nlu任务的形式，并生成intent和slot的vocab文件

（3）运行 task/crosswoz_nlu.py 文件进行训练、评估、推理等。







