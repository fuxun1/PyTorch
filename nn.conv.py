#卷积神经网络
import torch
import torch.nn.functional as F

#输入数据：二维矩阵
#原始图像
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]
                    ])

#卷积核
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]
                       ])

#卷积函数的输入图像的尺寸必须是四个参数，上述原始图像尺寸只有两个参数（5×5），所以要进行变换
#简单来说就是卷积函数的输入必须是四维张量
input = torch.reshape(input,(1,1,5,5))  # batch_size=1,通道=1
kernel = torch.reshape(kernel,(1,1,3,3))

#调用卷积函数进行卷积
output = F.conv2d(input,kernel,stride=1,padding=0)
print(output)

output2 = F.conv2d(input,kernel,stride=2,padding=0)
print(output2)