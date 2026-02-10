#最大池化操作
#目的：在尽量不丢失“关键信息”的前提下，压缩特征图，让模型更稳、更快、更泛化
#在CNN的池化操作里，最常用、最经典的就是：最大池化（MaxPool）
"""两类输入：矩阵和官方数据集，分别查看效果，官方数据集的池化输出结果用tensorboard查看,
因为最大池化操作就是提取重要信息，所以结果显示图片几乎成了马赛克"""
import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                     transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]], dtype=torch.float32)
#池化层input输入参数为N,C,H,W
input = torch.reshape(input,(-1,1,5,5))

#搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1=nn.MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

#实例化神经网络模型
tudui = Tudui()
#将数据输入到模型得到输出
output = tudui(input)
print(output)

writer=SummaryWriter("logs_maxpool")
step = 0
for data in dataloader:
    imgs, labels = data
    output = tudui(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()

