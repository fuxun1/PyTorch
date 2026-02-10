#通过实战搭建CIFAR10模型结构来学会Sequential的使用，简化代码
#查询资料得知该模型其实就是不断卷积+最大池化，最后经过2个线性层得到的
#注意：卷积函数的padding和stride参数需要根据卷积前后的通道数及尺寸代入公式计算得到，dilation默认为1
from torch import nn
import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


#神经网络模型结构：3轮卷积+最大池化，得到的结果展平，再2轮线性层
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        #搭建模型（方法1）
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.flatten = nn.Flatten()     #展平
        # self.linear1 = nn.Linear(in_features=1024, out_features=64)
        # self.linear2 = nn.Linear(in_features=64, out_features=10)

        #搭建模型（方法2），这种方法无需为每一层神经网络单独命名
        self.model1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )



    def forward(self, x):   #x不断覆盖
        # x = self.conv1(x)
        # ...
        # x = self.linear2(x)
        x = self.model1(x)  #一步到位
        return x

#实例化神经网络模型
tudui = Tudui()
print(tudui)

#测试模型搭建是否正确
input = torch.ones((64,3,32,32))    #全是1的图片
output = tudui(input)
print(output.shape)     # 输出torch.Size([64, 10])则搭建正确，报错则模型搭建有误

#画出模型结构图
writer = SummaryWriter("logs_seq")
writer.add_graph(tudui, input)
writer.close()
