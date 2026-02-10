#一些常用的损失函数的使用
#绝对误差、均方误差、交叉熵损失

import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([1,2,5],dtype=torch.float32)

loss1 = L1Loss(reduction='mean')    #默认计算方式就是mean，所以其实可不写
loss2 = L1Loss(reduction='sum')
loss_mse = MSELoss(reduction='mean')

result1 = loss1(input, target)
result2 = loss2(input, target)
result_mse = loss_mse(input, target)

print(result1)
print(result2)
print(result_mse)

x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))    #输入到交叉熵损失函数的数据有一定格式要求，这里进行转化(N,class)
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)