#利用nn.Module搭建神经网络的框架
import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

    def forward(self, input):
        output = input + 1
        return output

#上述继承nn.Module类创建的子类就相当于搭建了自己的神经网络模板/框架
#但是真正要用还得实例化对象，这才是真正创建了自己的神经网络
tudui = Tudui()     #实例化对象
x=torch.tensor(1.0)
output=tudui(x)
print(output)