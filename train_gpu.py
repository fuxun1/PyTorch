#以方式2为例

#完整的模型训练流程
"""包括下载数据、加载数据、搭建网络/导入网络模型、创建模型、定义损失函数、
    创建优化器、开始训练、反向传播、梯度更新、输出相关结果、保存模型、
    利用tensorboard查看训练过程、验证模型、计算分类准确率等等"""
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import *

#定义训练的设备
device = torch.device("cuda")

#准备数据集（训练数据集和测试数数据）
train_data = torchvision.datasets.CIFAR10(root="F:\python_and_AI\Hands-On PyTorch\data",
                        train=True,transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="F:\python_and_AI\Hands-On PyTorch\data",
                        train=False,transform=transforms.ToTensor(), download=True)
#获得数据集长度
train_size = len(train_data)
test_size = len(test_data)
print(f"训练数据集的长度为：{train_size}")
print(f"测试数据集的长度为：{test_size}")

#加载数据集
train_dataloader = torch.utils.data.DataLoader(dataset=train_data,batch_size=64)
test_dataloader = torch.utils.data.DataLoader(dataset=test_data,batch_size=64)

"""搭建神经网络
一般习惯是直接把模型部分单独放在一个.py文件再导入，所以这部分注释掉
两个文件必须放在同一个目录底下
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x            """

#创建模型
tudui = Tudui()
tudui.to(device)

#定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#创建优化器
learning_rate = 1e-2      # 学习率为1*10^(-2)即0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#训练的轮数
epoch = 10

#添加tensorboard
writer = SummaryWriter("logs_train")


for i in range(epoch):
    print(f"------第{i+1}轮训练开始------")
#开始训练
    tudui.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = tudui(imgs)
        loss = loss_fn(output, targets)
        #优化
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}，Loss：{loss.item()}")    #item()会把tensor数据类型转换为数字
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #每轮训练结束，对模型在测试集上进行测试
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0  #总的预测正确的个数，初始化为0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = tudui(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1)==targets).sum()
            total_accuracy += accuracy
    print(f"整体测试集上的Loss：{total_test_loss}")
    print(f"整体测试集上的正确率{total_accuracy / test_size}")
    writer.add_scalar("test_loss", total_test_loss, i)
    writer.add_scalar("test_accuracy", total_accuracy / test_size, i)

    #保存模型
    torch.save(tudui,"tudui_{}.pth".format(i+1))
    print(f"tudui_{i+1}.pth模型已保存")


