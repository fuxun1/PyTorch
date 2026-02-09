#构建一层卷积层的模型

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

#下载官方数据集CIFAR10的测试集
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                     transform=transforms.ToTensor())
#构建数据加载器
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=64)

#构建神经网络模型框架
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        #构建卷积层conv1
        #这里不区分第几层，真正决定“第几层”的是forward()，最先算的就是第一层...
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=3, stride=1, padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x

#实例化自己的模型
tudui = Tudui()

#用tensorboard查看
writer=SummaryWriter("conv")
step = 0
#对数据进行卷积
for data in dataloader:
    imgs, labels = data
    output = tudui(imgs)
    # print(imgs.shape)     #查看输入图片格式，torch.Size([64,3,32,32])
    # print(output.shape)   #查看输出图片格式，torch.Size([64,6,30,30])

    writer.add_images("input", imgs, step)
    #6通道图像无法显示，所以对其进行Reshape，多的重叠部分变平铺
    output = torch.reshape(output, (-1,3,30,30))     # -1会自动计算batch_size,相当于把多的通道平铺,batch_size增大
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

