#通过ImageNet数据集验证VGG模型的pretrained下载参数，查看当该下载参数取True/False时，模型参数的区别
#对vgg_16进行迁移学习
#ImageNet内置数据集的使用需要安装scipy库
"""设为True时，会下载预训练权重文件，就是把保存的有各个层的参数权重的文件下载下来，占用内存空间
    设为False时，随机初始化参数，不会下载文件占用空间，而只是加载模型而已
    简单来说，一个是训练好的预训练模型，一个是还没训练的模型"""
import torchvision
from torch import nn

#设为False
vgg16_false = torchvision.models.vgg16(pretrained=False)
#设为True
vgg16_true = torchvision.models.vgg16(pretrained=True)

print('ok')
print(vgg16_true)   #查看网络架构

"""该模型输出类别为1000个类，如果要应用到10个类别的数据集上，就要用到迁移学习
    VGG_16作为前置的神经网络，在此基础上进行迁移学习得到自己想要的"""
#对现有模型进行修改--迁移学习
#为vgg16_true在最后添加一层线性层add_linear，实现从输出1000个类别到10个类别
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)   #查看模型结构
#修改vgg16_false的第7层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_true)   #查看模型结构
#删除最后一层
vgg16_true.classifier = nn.Sequential(
    *list(vgg16_true.classifier.children())[:-1]
)   # 原理：拿到所有层，去掉最后一层，重新包成Sequential