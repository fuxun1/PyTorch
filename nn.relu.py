#构建一个神经网络，拥有一层relu非线性激活层
import torch
from torch import nn

input = torch.tensor([[1,-0.5],
                    [-1,3]])
input = torch.reshape(input,[-1,1,2,2])

#搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = nn.ReLU()      #inplace参数，决定是否提前覆盖原始输入数据，默认且建议为False

    def forward(self,input):
        output = self.relu1(input)
        return output

#实例化神经网络模型
tudui = Tudui()
output = tudui(input)
print(output)