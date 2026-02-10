#学会优化器的使用及模型的完整训练流程

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torch import nn

dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                     transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

tudui = Tudui()

loss = CrossEntropyLoss()
#设置优化器
#采用随机梯度下降（SGD）
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
#设置训练20轮来查看这20轮总误差的变化情况
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        optim.zero_grad()   #梯度清零
        outputs = tudui(imgs)   #前向传播
        result_loss = loss(outputs, targets)    #计算损失
        result_loss.backward()  #反向传播（计算梯度）
        optim.step()    #更新参数
        #对损失进行累加
        running_loss = running_loss + result_loss
    print(running_loss)

