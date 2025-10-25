# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数类：用于存储模型训练所需的所有超参数和路径"""

    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'  # 当前模型名称

        # 数据集路径：训练、验证、测试文件均放在 THUCNNews/data/ 目录下
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集

        # 类别名单：从 class.txt 中读取每一行作为类别名（如 '体育', '财经' 等）
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单

        # 词表路径：使用 pickle 保存的词汇表（词到索引的映射）
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表

        # 模型保存路径：训练完成后保存模型参数（.ckpt 文件）
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果

        # 日志路径：用于 TensorBoard 可视化训练过程
        self.log_path = dataset + '/log/' + self.model_name

        # 预训练词向量：如果提供了 embedding 文件（如 embedding_SougouNews.npz），则加载；
        # 否则为 None，表示随机初始化
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量

        # 设备设置：自动选择 GPU（如果有）或 CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        # Dropout 比率：防止过拟合，在训练时随机丢弃部分神经元
        self.dropout = 0.5

        # 提前停止阈值：若连续 1000 个 batch 没有在验证集上提升，则提前终止训练
        self.require_improvement = 1000

        # 分类类别数量：由 class.txt 文件中的类别数量决定
        self.num_classes = len(self.class_list)

        # 词表大小：初始化为 0，将在数据预处理阶段根据 vocab 填充
        self.n_vocab = 0

        # 训练轮数
        self.num_epochs = 10

        # 每个 batch 的样本数量
        self.batch_size = 128

        # 序列长度统一化：所有句子会被截断或补零到 20 个 token
        self.pad_size = 20

        # 学习率：优化器（如 Adam）使用的步长
        self.learning_rate = 1e-3

        # 词向量维度：
        # 如果使用预训练词向量，则维度由预训练向量决定；
        # 否则默认为 300
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300

        # LSTM 隐藏层维度：每个时间步隐藏状态的大小
        self.hidden_size = 128

        # LSTM 层数：堆叠的 LSTM 层数量
        self.num_layers = 3


# 这是一个经典的基于 RNN 的文本分类模型，使用双向 LSTM 提取序列特征


class Model(nn.Module):
    """TextRNN 模型：基于双向 LSTM 的文本分类网络"""

    def __init__(self, config):
        super(Model, self).__init__()
        # 初始化词嵌入层
        if config.embedding_pretrained is not None:
            # 如果有预训练词向量，使用它初始化嵌入层，并允许微调（freeze=False）
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            # 否则随机初始化词嵌入层，词表大小为 n_vocab，维度为 embed
            # padding_idx 表示填充 token 的索引，在训练时其梯度不会更新
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        # 定义双向 LSTM 层：
        # 输入维度：词向量维度 (config.embed)
        # 隐藏层维度：128
        # 层数：3 层堆叠 LSTM
        # bidirectional=True：使用双向 LSTM，能同时捕捉前后文信息
        # batch_first=True：输入输出张量格式为 (batch_size, seq_len, feature)
        # dropout：在除最后一层外的每层后应用 dropout（最后一层无 dropout）
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)

        # 全连接层：将 LSTM 最后时刻的输出（双向拼接后为 2*hidden_size）映射到类别空间
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入数据，形式为 (tokens, lengths)，其中：
                  - tokens: [batch_size, seq_len]，词索引序列
                  - lengths: [batch_size]，原始序列长度（用于处理变长序列）
        :return: 分类 logits [batch_size, num_classes]
        """
        # 解包输入，这里只使用 tokens，忽略 lengths（实际中可用于 pack_padded_sequence）
        x, _ = x

        # 词嵌入：将词索引转换为向量表示
        # 输出形状：[batch_size, seq_len, embedding_dim]，例如 [128, 20, 300]
        out = self.embedding(x)

        # 将嵌入输入 LSTM
        # 输出 out: [batch_size, seq_len, hidden_size * 2]（双向所以是 2×hidden_size）
        # _ 包含 (h_n, c_n)，即最终的隐藏状态和细胞状态
        out, _ = self.lstm(out)

        # 取最后一个时间步的输出作为整个序列的表示
        # 因为是分类任务，我们假设最后时刻的 hidden state 包含了完整的句子语义
        # out[:, -1, :] 的形状为 [batch_size, hidden_size * 2]
        out = self.fc(out[:, -1, :])  # 形状变为 [batch_size, num_classes]

        # 返回未归一化的预测分数（logits），后续配合 CrossEntropyLoss 使用
        return out