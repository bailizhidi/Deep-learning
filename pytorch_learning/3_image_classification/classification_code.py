# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# 设置数据路径
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl"  # 已下载的文件名

# 确保数据目录存在
PATH.mkdir(parents=True, exist_ok=True)

# 加载已下载的mnist.pkl文件
print("正在加载已下载的mnist.pkl文件...")
with open(PATH / FILENAME, 'rb') as f:
    # 读取数据（注意：原始mnist.pkl包含训练集、验证集和测试集）
    ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding='latin1')

# 将数据转换为PyTorch张量
print("转换数据为PyTorch张量...")
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_valid = torch.tensor(x_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 打印数据形状
print(f"训练集形状: {x_train.shape}, 标签形状: {y_train.shape}")
print(f"验证集形状: {x_valid.shape}, 标签形状: {y_valid.shape}")
print(f"测试集形状: {x_test.shape}, 标签形状: {y_test.shape}")

# 创建数据集和数据加载器
print("创建数据集和数据加载器...")
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
test_ds = TensorDataset(x_test, y_test)

batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2)


# 定义神经网络模型
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层: 784个像素 (28x28)
        # 隐藏层1: 128个神经元
        # 隐藏层2: 256个神经元
        # 输出层: 10个类别 (0-9)
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        # 前向传播: 输入 -> 隐藏层1 -> ReLU -> 隐藏层2 -> ReLU -> 输出层
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


# 创建模型实例
print("创建模型...")
model = Mnist_NN()

# 定义损失函数和优化器
loss_func = F.cross_entropy  # 交叉熵损失，适用于分类问题
opt = optim.Adam(model.parameters(), lr=0.001)  # 随机梯度下降优化器


# 训练和评估函数
def loss_batch(model, loss_func, xb, yb, opt=None):
    """计算批次损失并更新参数"""
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()  # 计算梯度
        opt.step()  # 更新权重
        opt.zero_grad()  # 重置梯度

    return loss.item(), len(xb)


def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    """训练模型并打印验证损失"""
    for step in range(steps):
        # 训练模式
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            loss, _ = loss_batch(model, loss_func, xb, yb, opt)
            train_losses.append(loss)

        # 评估模式
        model.eval()
        with torch.no_grad():
            val_losses, nums = zip(*[loss_batch(model, loss_func, xb, yb)
                                     for xb, yb in valid_dl])
            val_loss = np.sum(np.multiply(val_losses, nums)) / np.sum(nums)

        # 打印训练进度
        print(f'Step {step + 1}/{steps} | 训练损失: {np.mean(train_losses):.4f} | 验证损失: {val_loss:.4f}')


# 开始训练
print("\n开始训练模型...")
fit(25, model, loss_func, opt, train_dl, valid_dl)

# 测试模型
print("\n测试模型性能...")
model.eval()
with torch.no_grad():
    test_losses, nums = zip(*[loss_batch(model, loss_func, xb, yb)
                              for xb, yb in test_dl])
    test_loss = np.sum(np.multiply(test_losses, nums)) / np.sum(nums)
    print(f"测试集损失: {test_loss:.4f}")

# 计算准确率
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_dl:
        outputs = model(xb)
        _, predicted = torch.max(outputs.data, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")