#模型的加载

import torch
import torchvision
from torch import nn

"""加载模型"""

#对应模型保存方式1
#weights_only=False 的作用是：允许torch.load反序列化“完整的 Python对象”，而不只是权重张量
model1 = torch.load("F:\\python_and_AI\\Hands-On PyTorch\\vgg16_method1.pth",weights_only=False)
# print(model1)   #输出结果是模型的结构

#对应模型保存方式2（官方推荐）
model2 = torch.load("F:\\python_and_AI\\Hands-On PyTorch\\vgg16_method2.pth")
# print(model2)   #输出结果是字典的形式，显示的是参数，这还不是模型
vgg16 = torchvision.models.vgg16(pretrained=False)      #获得模型结构
vgg16.load_state_dict(model2)   #将加载的模型参数放入到模型结构中，获得所需模型
# print(vgg16)

#陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

    def forward(self,x):
        x=self.conv1(x)
        return x

model = torch.load("F:\\python_and_AI\\Hands-On PyTorch\\tudui.pth",weights_only=False)
print(model)