#将交叉熵损失函数应用到之前搭建的CIFAR10模型中，查看每个batch_size预测值与目标值之间的差距

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
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)

