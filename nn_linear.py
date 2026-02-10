#利用线性层，将5×5的图片展开为1×25，再变成1×3
#将图像展平，平铺成一行：reshape(input,[1,1,1,-1])

import torch
import torchvision
from setuptools.namespaces import flatten
from torch import nn
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.line1 = nn.Linear(196608,10)
    def forward(self,input):
        output = self.line1(input)
        return output

#实例化模型
tudui = Tudui()

for data in dataloader:
    imgs, labels = data
    print(imgs.shape)       #未展平的形状：torch.Size([64, 3, 32, 32])
    # output = torch.reshape(imgs,[1,1,1,-1])     #调整格式：展平（手动）
    output = torch.flatten(imgs)    #调整格式：展平（自动）
    print(output.shape)     #展平后的形状：torch.Size([1,1,1,196608]) / torch.Size([196608])
    output = tudui(output)
    print(output.shape)     #输出线性变换后的数据格式：torch.Size([1,1,1,10]) / torch.Size([10])
