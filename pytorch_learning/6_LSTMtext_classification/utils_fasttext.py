# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm  # 进度条显示工具
import time
from datetime import timedelta  # 时间差格式化


# 预定义常量
MAX_VOCAB_SIZE = 10000        # 词汇表最大容量
UNK, PAD = '<UNK>', '<PAD>'   # 未知词标记 和 填充标记


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    构建词汇表（词到索引的映射）
    :param file_path: 训练集文件路径
    :param tokenizer: 分词函数（char-level 或 word-level）
    :param max_size: 词汇表最大词数
    :param min_freq: 最小词频，低于此频率的词不加入词汇表
    :return: 字典 {word: index}
    """
    vocab_dic = {}  # 临时词频统计字典

    # 读取训练文件，逐行处理
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):  # 显示进度条
            lin = line.strip()  # 去除首尾空白字符
            if not lin:         # 跳过空行
                continue
            # 每行格式：文本内容 \t 标签
            content = lin.split('\t')[0]  # 取出文本部分
            # 使用 tokenizer 对文本分词，并更新词频
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 若不存在则初始化为0再+1

        # 筛选词频 >= min_freq 的词，并按频率降序排列，取前 max_size 个
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]

        # 将词映射为索引（从0开始）
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}

        # 添加特殊标记：<UNK> 表示未知词，<PAD> 表示填充
        # 它们的索引紧接在正常词之后
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    return vocab_dic


def build_dataset(config, ues_word):
    """
    构建训练、验证、测试数据集
    :param config: 配置对象，包含路径、pad_size等
    :param ues_word: 是否使用 word-level 分词（否则 char-level）
    :return: vocab词表，以及 train/dev/test 数据集
    """
    # 定义分词器
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格切分，适用于已分好词的文本
    else:
        tokenizer = lambda x: [y for y in x]  # 字符级分词，每个汉字作为一个token

    # 判断是否已有保存的词汇表文件
    if os.path.exists(config.vocab_path):
        # 如果存在，直接加载（避免重复构建）
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        # 否则从训练集构建词汇表并保存
        vocab = build_vocab(config.train_path, tokenizer=tokenizer,
                            max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))  # 保存为 .pkl 文件

    print(f"Vocab size: {len(vocab)}")  # 输出词汇表大小

    # ------------------------ N-Gram Hash 函数（用于 FastText 等模型）------------------------
    def biGramHash(sequence, t, buckets):
        """
        计算第 t 个位置的 bigram 哈希值
        :param sequence: 已转换为ID的序列
        :param t: 当前位置
        :param buckets: 哈希桶总数（即n-gram词表大小）
        :return: 哈希后的索引
        """
        # 获取前一个词的ID，若越界则用0（相当于<PAD>）
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets  # 大质数哈希，减少冲突

    def triGramHash(sequence, t, buckets):
        """
        计算第 t 个位置的 trigram 哈希值
        """
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    # ------------------------ 加载单个数据集（如 train/dev/test）------------------------
    def load_dataset(path, pad_size=32):
        """
        加载指定路径的数据集
        :param path: 数据文件路径
        :param pad_size: 统一序列长度
        :return: 列表，元素为 (tokens_id, label, seq_len, bigram, trigram)
        """
        contents = []  # 存储所有样本

        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):  # 显示进度条
                lin = line.strip()
                if not lin:       # 跳过空行
                    continue
                # 每行格式：文本 \t 标签
                content, label = lin.split('\t')

                words_line = []   # 存储当前句子的词ID序列
                token = tokenizer(content)  # 分词
                seq_len = len(token)  # 原始长度

                # --- 序列截断或填充 ---
                if pad_size:
                    if len(token) < pad_size:
                        # 不足则用 <PAD> 填充
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                    else:
                        # 超长则截断
                        token = token[:pad_size]
                        seq_len = pad_size  # 更新实际参与计算的长度

                # --- 词转ID ---
                for word in token:
                    # 若词在词表中，取其ID；否则用 <UNK>
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # --- 构造 n-gram 特征（用于某些模型如 FastText）---
                buckets = config.n_gram_vocab  # n-gram 哈希桶数量
                bigram = []
                trigram = []
                for i in range(pad_size):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))

                # --- 添加样本 ---
                # 注意：此时 words_line 是 ID 列表，label 转为整数
                contents.append((words_line, int(label), seq_len, bigram, trigram))

        return contents  # 返回列表：[(token_ids, label, length, bigram, trigram), ...]

    # 分别加载训练、验证、测试集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, dev, test  # 返回词表和三个数据集


# ------------------------ 自定义数据迭代器 ------------------------
class DatasetIterater(object):
    """
    数据加载迭代器，支持 batch 批量生成
    """

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size  # 完整 batch 的数量
        self.residue = False  # 是否有剩余不足一个 batch 的数据
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0  # 当前 batch 索引
        self.device = device  # 数据放置的设备（CPU/GPU）

    def _to_tensor(self, datas):
        """
        将一个 batch 的数据转换为 PyTorch Tensor
        :param datas: 一个 batch 的原始数据列表
        :return: ((x, seq_len, bigram, trigram), y)
        """
        # 提取各字段并转换为 LongTensor（适合嵌入层输入）
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)          # token IDs
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)          # labels
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)     # bigram features
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)    # trigram features
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)    # 实际序列长度

        return (x, seq_len, bigram, trigram), y  # 返回特征元组和标签

    def __next__(self):
        """
        实现迭代器协议：返回下一个 batch
        """
        # 处理最后一个不完整的 batch（残差）
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            return self._to_tensor(batches)

        # 所有 batch 已遍历完，重置并抛出 StopIteration
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration

        # 正常 batch
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return self._to_tensor(batches)

    def __iter__(self):
        return self

    def __len__(self):
        """
        返回总 batch 数
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """
    创建数据迭代器
    :param dataset: 数据集（如 train/dev/test）
    :param config: 配置对象
    :return: DatasetIterater 实例
    """
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """
    计算并格式化耗时
    :param start_time: 开始时间戳（time.time()）
    :return: timedelta 对象，如 "1min 23s"
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# ======================== 主程序入口：提取预训练词向量 ========================
if __name__ == "__main__":
    '''提取预训练词向量'''
    vocab_dir = "./THUCNews/data/vocab.pkl"           # 本地词汇表路径
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"  # 预训练中文词向量文件（搜狗字符级）
    emb_dim = 300                                     # 词向量维度
    filename_trimmed_dir = "./THUCNews/data/vocab.embedding.sougou"  # 输出的裁剪后词向量文件

    # 加载本地词汇表
    word_to_id = pkl.load(open(vocab_dir, 'rb'))

    # 初始化嵌入矩阵：随机初始化所有词向量
    embeddings = np.random.rand(len(word_to_id), emb_dim)

    # 读取预训练词向量文件（文本格式：每行 一个词 + 300维向量）
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # 示例行："中国 0.123 0.456 ... 0.789"
        lin = line.strip().split(" ")
        # 第一个元素是词
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]  # 获取该词在本地词表中的索引
            # 取出前300个浮点数作为词向量（防止多余列干扰）
            emb = [float(x) for x in lin[1:301]]
            # 赋值到 embedding 矩阵对应行
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()

    # 将匹配好的词向量矩阵压缩保存为 .npz 文件
    # 可通过 np.load() 读取，字段名为 'embeddings'
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

    print(f"Pretrained embeddings saved to {filename_trimmed_dir}")