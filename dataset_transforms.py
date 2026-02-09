import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#设置转化形式
dataset_transform=transforms.Compose(
        [transforms.ToTensor()
         ])
#下载数据集并将转化形式作为参数传入
#将图片类型由PIL转化为tensor
train_set=torchvision.datasets.CIFAR10(root="F:\python_and_AI\Hands-On PyTorch",
 train=True, transform=dataset_transform, download=True)
test_set=torchvision.datasets.CIFAR10(root="F:\python_and_AI\Hands-On PyTorch",
 train=False, transform=dataset_transform, download=True)

# print(test_set[0])  #输出二元组，可见一个数据集元素是由(图片，类别)组成
# print(test_set.classes)  #输出测试集的所有类别
# img,targets=test_set[0]
# print(img)
# print(targets)
# print(test_set.classes[targets])
# img.show()
print(test_set[0])
writer=SummaryWriter("test_set")
for i in range(10):
   img,target=test_set[i]
   writer.add_image("test_set",img,i)

