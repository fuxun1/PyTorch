#对训练得到的模型进行测试

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn

img_path = "F:\\python_and_AI\\dog.jpg"
img = Image.open(img_path)
#模型输入格式要求为32*32
transform = transforms.Compose([
    transforms.Resize((32,32)),
    torchvision.transforms.ToTensor(),
])
img = transform(img)

#设备
#因为模型是在gpu训练得到的，所以这里输入数据也需要送到gpu


#加载模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x
model = torch.load("tudui_4.pth", map_location="cpu")

#转换格式
img = torch.reshape(img, (1,3,32,32))

#开始验证
model.eval()
with torch.no_grad():
    output = model(img)
    pre_class = output.argmax(1)
print(pre_class)        # 期望输出：tensor([5], device='cuda:0')