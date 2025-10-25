# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm  # 显示循环进度条的工具
import time
from datetime import timedelta  # 用于格式化时间差（如“0:01:30”）


# 定义全局常量
MAX_VOCAB_SIZE = 10000  # 词汇表最大词数（只保留最频繁的10000个词）
UNK, PAD = '<UNK>', '<PAD>'  # 特殊标记：<UNK>表示未知词，<PAD>表示填充符


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    根据训练集文本构建词汇表（词到索引的映射）
    :param file_path: 训练数据文件路径（如 train.txt）
    :param tokenizer: 分词函数，决定是按字还是按词切分
    :param max_size: 词汇表最大容量
    :param min_freq: 最小词频，低于此频率的词不加入词表
    :return: 字典 {word: index}
    """
    vocab_dic = {}  # 临时词频统计字典

    # 打开训练文件，逐行读取
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):  # 使用tqdm显示处理进度
            lin = line.strip()  # 去除首尾空白字符（换行符、空格等）
            if not lin:        # 跳过空行
                continue
            # 每行格式：文本内容 \t 标签（例如：“今天天气很好\t1”）
            content = lin.split('\t')[0]  # 取出文本部分（标签在后面）

            # 使用 tokenizer 对文本进行分词（可能是按字或按词）
            for word in tokenizer(content):
                # 统计每个词出现的次数
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

        # 筛选出词频 >= min_freq 的词，并按频率从高到低排序，取前 max_size 个
        vocab_list = sorted(
            [item for item in vocab_dic.items() if item[1] >= min_freq],
            key=lambda x: x[1], reverse=True
        )[:max_size]

        # 将词映射为索引（从0开始）
        vocab_dic = {word: idx for idx, (word, freq) in enumerate(vocab_list)}

        # 添加特殊标记 <UNK> 和 <PAD>，索引紧接在正常词之后
        vocab_dic.update({
            UNK: len(vocab_dic),      # <UNK> 索引为当前长度
            PAD: len(vocab_dic) + 1   # <PAD> 再往后一位
        })

    return vocab_dic  # 返回最终的词汇表字典


def build_dataset(config, ues_word):
    """
    构建训练、验证、测试数据集
    :param config: 配置对象，包含路径、pad_size等参数
    :param ues_word: 是否使用 word-level 分词（否则使用 char-level）
    :return: vocab词表，以及 train/dev/test 数据集（均为列表）
    """
    # 定义分词方式
    if ues_word:
        # 按词切分：假设输入文本中词之间用空格隔开（如“我 爱 北京 天安门”）
        tokenizer = lambda x: x.split(' ')
    else:
        # 按字符切分：将每个汉字作为一个token（如“你好” → ['你', '好']）
        tokenizer = lambda x: [y for y in x]

    # 判断是否已有保存好的词汇表文件
    if os.path.exists(config.vocab_path):
        # 如果存在，直接加载（避免重复构建）
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        # 否则从训练集重新构建词汇表
        vocab = build_vocab(
            config.train_path,
            tokenizer=tokenizer,
            max_size=MAX_VOCAB_SIZE,
            min_freq=1  # 最小频率为1，即只出现一次的词也保留
        )
        # 将构建好的词汇表保存到本地，下次可直接加载
        pkl.dump(vocab, open(config.vocab_path, 'wb'))

    print(f"Vocab size: {len(vocab)}")  # 输出词汇表总大小（含特殊标记）

    def load_dataset(path, pad_size=32):
        """
        加载指定路径的数据集（训练/验证/测试）
        :param path: 数据文件路径
        :param pad_size: 统一的序列长度（padding后长度）
        :return: 列表，每个元素是一个样本：(token_ids, label, seq_len)
        """
        contents = []  # 存放所有样本

        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):  # 显示加载进度
                lin = line.strip()
                if not lin:       # 跳过空行
                    continue

                # 解析每一行：文本 \t 标签
                content, label = lin.split('\t')

                words_line = []   # 存放当前句子的词ID序列
                token = tokenizer(content)  # 分词
                seq_len = len(token)  # 原始长度

                # --- 序列长度处理：padding 或 截断 ---
                if pad_size:
                    if len(token) < pad_size:
                        # 不足 pad_size：用 <PAD> 填充到指定长度
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                    else:
                        # 超过 pad_size：截断前面部分
                        token = token[:pad_size]
                        seq_len = pad_size  # 更新实际参与计算的长度

                # --- 将词转换为对应的ID ---
                for word in token:
                    # 如果词在词表中，取其ID；否则用 <UNK> 的ID
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # --- 添加样本到数据集中 ---
                # 每个样本包含：[token_ids列表, 标签, 实际长度]
                contents.append((words_line, int(label), seq_len))

        return contents  # 返回处理好的数据列表

    # 分别加载训练集、验证集、测试集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    # 返回词汇表和三个数据集
    return vocab, train, dev, test


class DatasetIterater(object):
    """
    自定义数据迭代器，用于按 batch 返回数据
    支持 GPU 加速（自动将数据放到指定 device 上）
    """

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches           # 所有样本的列表
        self.n_batches = len(batches) // batch_size  # 完整 batch 的数量
        self.residue = False             # 是否有剩余不足一个 batch 的数据
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0                   # 当前 batch 的索引
        self.device = device             # 数据要送到的设备（'cpu' 或 'cuda'）

    def _to_tensor(self, datas):
        """
        将一个 batch 的原始数据转换为 PyTorch Tensor
        :param datas: 当前 batch 的样本列表
        :return: ((input_ids, seq_len), labels)
        """
        # 提取 input_ids 并转为 LongTensor（嵌入层需要 long 类型）
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # 提取标签
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # 提取原始序列长度（用于 RNN 类模型控制序列长度）
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        return (x, seq_len), y  # 返回特征元组和标签

    def __next__(self):
        """
        迭代器核心方法：返回下一个 batch
        """
        # 情况1：还有最后一个不完整的 batch（残差）
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            return self._to_tensor(batches)

        # 情况2：所有 batch 都已遍历完，重置并停止迭代
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration

        # 情况3：正常返回一个完整 batch
        else:
            start = self.index * self.batch_size
            end = (self.index + 1) * self.batch_size
            batches = self.batches[start:end]
            self.index += 1
            return self._to_tensor(batches)

    def __iter__(self):
        return self

    def __len__(self):
        """
        返回总的 batch 数量
        """
        if self.residue:
            return self.n_batches + 1  # 有残差则多一个 batch
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """
    创建一个可迭代的数据加载器
    :param dataset: 数据集（train/dev/test）
    :param config: 配置对象（包含 batch_size 和 device）
    :return: DatasetIterater 实例
    """
    return DatasetIterater(dataset, config.batch_size, config.device)


def get_time_dif(start_time):
    """
    计算程序运行时间，并格式化输出
    :param start_time: 开始时间戳（time.time()）
    :return: timedelta 对象，例如 "0:01:23"
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# ======================== 主程序入口：提取预训练词向量 ========================
if __name__ == "__main__":
    '''提取并裁剪预训练词向量（如搜狗新闻字符级向量）'''
    # 路径配置（根据实际情况修改）
    train_dir = "./THUCNews/data/train.txt"           # 训练集路径
    vocab_dir = "./THUCNews/data/vocab.pkl"           # 本地词汇表保存路径
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"  # 预训练词向量文件路径（字符级）
    emb_dim = 300                                     # 词向量维度（搜狗是300维）
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"  # 输出文件路径

    # --- 第一步：构建或加载词汇表 ---
    if os.path.exists(vocab_dir):
        # 若已有词表，直接加载
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # 否则重新构建词表（这里使用字符级分词）
        # tokenizer = lambda x: x.split(' ')  # 如果是词级，取消注释这行
        tokenizer = lambda x: [y for y in x]  # 字符级分词
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 保存新构建的词表
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    # --- 第二步：初始化嵌入矩阵 ---
    # 创建随机初始化的 embedding 矩阵：[vocab_size, emb_dim]
    embeddings = np.random.rand(len(word_to_id), emb_dim)

    # --- 第三步：加载预训练向量并匹配 ---
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # 每行格式：词 + 300个浮点数（空格分隔）
        lin = line.strip().split(" ")
        word = lin[0]  # 第一个元素是词

        # 如果这个词在我们的本地词表中存在
        if word in word_to_id:
            idx = word_to_id[word]  # 获取它在本地词表中的索引
            # 取出对应的300维向量（防止多余列，只取前300个）
            emb = [float(x) for x in lin[1:301]]
            # 赋值到 embedding 矩阵中
            embeddings[idx] = np.asarray(emb, dtype='float32')

    f.close()  # 关闭文件

    # --- 第四步：保存裁剪后的词向量 ---
    # 使用 np.savez_compressed 压缩保存为 .npz 文件
    # 可通过 np.load('xxx.npz')['embeddings'] 读取
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

    print(f"预训练词向量已提取并保存至: {filename_trimmed_dir}.npz")
    print(f"词表大小: {len(word_to_id)}, 词向量维度: {emb_dim}")