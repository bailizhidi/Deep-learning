# 导入依赖包
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter # 导入SummaryWriter，用于将训练过程中的损失、准确率等指标写入日志，供TensorBoard可视化工具查看

# --model TextRNN
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# 添加必需参数 --model，用户必须指定一个模型名称（字符串），用于选择要使用的模型架构
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# 添加可选参数 --embedding，默认值为 'pre_trained'，表示使用预训练词向量；也可以设为 'random' 表示随机初始化词向量
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# 添加参数 --word，控制是否以“词”为单位进行分词（True）还是以“字符”为单位（False）
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    # 如果用户指定 --embedding random，则将 embedding 设为 'random'，表示不使用预训练词向量，而是随机初始化
    if args.embedding == 'random':
        embedding = 'random'
    #获取用户选择的模型名
    model_name = args.model  #TextCNN, TextRNN,
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        # 强制使用随机词向量（因为 FastText 自身会学习 subword 信息，通常不需要预训练向量）
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    # 调用所选模型模块中的 Config 类
    config = x.Config(dataset, embedding)
    # NumPy随机种子 CPU上的PyTorch随机种子 所有GPU上的随机种子（适用于多卡）
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 设置 CuDNN 为确定性模式，保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # 调用 build_dataset 函数
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    # 使用 build_iterator 将原始数据转换为 PyTorch DataLoader 迭代器
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # 将词汇表大小写入配置对象，供模型定义词嵌入层时使用
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    # 创建 SummaryWriter 实例，用于写入 TensorBoard 日志
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    # 开始训练
    train(config, model, train_iter, dev_iter, test_iter,writer)
