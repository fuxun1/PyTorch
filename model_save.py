#模型的保存

import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

#保存方式1
torch.save(vgg16, "F:\\python_and_AI\\Hands-On PyTorch\\vgg16_method1.pth")
#保存方式2（官方推荐）
#只保存参数，以字典的形式保存，节省空间
torch.save(vgg16.state_dict(),"F:\\python_and_AI\\Hands-On PyTorch\\vgg16_method2.pth")

#陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

    def forward(self,x):
        x=self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui,"F:\\python_and_AI\\Hands-On PyTorch\\tudui.pth")