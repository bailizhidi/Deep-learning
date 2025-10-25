# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics  # 用于计算准确率、分类报告、混淆矩阵等指标
import time
from utils import get_time_dif  # 自定义工具函数：格式化时间差（如 1min 23s）
from tensorboardX import SummaryWriter  # 用于将训练日志写入文件，供 TensorBoard 可视化


# 权重初始化函数，默认使用 Xavier 初始化
def init_network(model, method='xavier', exclude='embedding', seed=123):
    """
    对模型参数进行初始化，跳过指定层（如 embedding 层）
    :param model: 要初始化的模型
    :param method: 初始化方法，支持 'xavier'、'kaiming' 或 'normal'
    :param exclude: 不进行初始化的层名关键字（如 embedding 通常不初始化或单独处理）
    :param seed: 随机种子，保证可复现性
    """
    # 设置全局随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # 遍历模型所有可训练参数（包括名称和参数张量）
    for name, w in model.named_parameters():
        # 如果该参数名称包含 exclude（如 'embedding'），则跳过不初始化
        if exclude not in name:
            if 'weight' in name:  # 权重参数
                if method == 'xavier':
                    nn.init.xavier_normal_(w)  # Xavier 初始化，适合 Sigmoid/Tanh 激活函数
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)  # Kaiming 初始化，适合 ReLU 激活函数
                else:
                    nn.init.normal_(w)  # 正态分布初始化
            elif 'bias' in name:  # 偏置参数
                nn.init.constant_(w, 0)  # 偏置初始化为 0
            else:
                pass  # 其他情况不处理


def train(config, model, train_iter, dev_iter, test_iter, writer):
    """
    训练主函数
    :param config: 配置对象，包含超参数和路径
    :param model: 要训练的模型
    :param train_iter: 训练集 DataLoader
    :param dev_iter: 验证集 DataLoader
    :param test_iter: 测试集 DataLoader
    :param writer: TensorBoard 日志记录器
    """
    start_time = time.time()
    model.train()  # 切换模型为训练模式（启用 dropout、batchnorm 等）

    # 定义优化器：使用 Adam 优化器，学习率来自配置
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # （可选）学习率衰减策略：每轮 epoch 后乘以 gamma 降低学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_batch = 0  # 记录当前训练到了第几个 batch
    dev_best_loss = float('inf')  # 记录验证集上的最小 loss
    last_improve = 0  # 记录上次验证集 loss 下降时的 batch 数
    flag = False  # 是否触发“长时间无提升”而提前停止

    # 开始训练循环（按 epoch）
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        # （可选）每轮 epoch 结束后衰减学习率
        # scheduler.step()

        # 遍历训练集每个 batch
        for i, (trains, labels) in enumerate(train_iter):
            # trains: (tokens, lengths) 或直接是 tokens，取决于数据迭代器实现
            # labels: [batch_size]，真实类别标签

            outputs = model(trains)           # 前向传播，得到预测 logits
            model.zero_grad()                 # 梯度清零
            loss = F.cross_entropy(outputs, labels)  # 计算交叉熵损失
            loss.backward()                   # 反向传播，计算梯度
            optimizer.step()                  # 更新模型参数

            # 每 100 个 batch 输出一次训练和验证结果
            if total_batch % 100 == 0:
                # 计算训练集准确率
                true = labels.data.cpu()  # 真实标签
                predic = torch.max(outputs.data, 1)[1].cpu()  # 预测类别（最大 logit 的索引）
                train_acc = metrics.accuracy_score(true, predic)

                # 在验证集上评估模型性能
                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                # 如果验证集 loss 降低，保存当前最优模型
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)  # 仅保存模型参数
                    improve = '*'  # 标记为性能提升
                    last_improve = total_batch  # 更新最后一次提升的 batch 数
                else:
                    improve = ''  # 无提升

                # 计算并格式化训练耗时
                time_dif = get_time_dif(start_time)
                # 打印当前训练状态信息
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                # 将指标写入 TensorBoard 日志
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

                model.train()  # 再次切换回训练模式（因为 evaluate 中会切换为 eval）

            total_batch += 1  # batch 计数 +1

            # 如果超过 config.require_improvement 个 batch 验证 loss 未下降，提前停止
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True  # 触发提前停止
                break  # 跳出当前 epoch 的 batch 循环
        if flag:
            break  # 跳出整个 epoch 循环

    writer.close()  # 关闭 TensorBoard 写入器
    test(config, model, test_iter)  # 使用最优模型在测试集上评估


def test(config, model, test_iter):
    """
    在测试集上评估模型性能
    :param config: 配置对象
    :param model: 模型
    :param test_iter: 测试集 DataLoader
    """
    # 加载训练过程中保存的最佳模型参数
    model.load_state_dict(torch.load(config.save_path))

    model.eval()  # 切换为评估模式（关闭 dropout，冻结 batchnorm）
    start_time = time.time()

    # 调用 evaluate 函数，返回测试集上的各项指标
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    # 打印测试损失和准确率
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))

    # 打印详细的分类报告（精确率、召回率、F1 值）
    print("Precision, Recall and F1-Score...")
    print(test_report)

    # 打印混淆矩阵
    print("Confusion Matrix...")
    print(test_confusion)

    # 打印测试耗时
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    """
    评估模型在给定数据集上的性能
    :param config: 配置对象
    :param model: 模型
    :param data_iter: 数据迭代器（验证集或测试集）
    :param test: 是否为最终测试（决定是否返回详细报告）
    :return: 准确率、平均损失、（可选）分类报告、混淆矩阵
    """
    model.eval()  # 评估模式
    loss_total = 0  # 累计损失
    predict_all = np.array([], dtype=int)  # 存储所有预测标签
    labels_all = np.array([], dtype=int)   # 存储所有真实标签

    # 不计算梯度，节省内存和计算
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)  # 前向传播
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()  # 累加损失值

            # 将预测结果和真实标签转移到 CPU 并转为 numpy 数组
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # 取最大 logit 的类别

            # 拼接到全局数组中
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)

    # 如果是测试阶段，生成详细报告
    if test:
        report = metrics.classification_report(
            labels_all,
            predict_all,
            target_names=config.class_list,  # 类别名称
            digits=4  # 保留 4 位小数
        )
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion

    # 验证阶段只需返回准确率和平均损失
    return acc, loss_total / len(data_iter)