import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


dataset_transform=transforms.Compose(
        [transforms.ToTensor()
         ])

test_data=torchvision.datasets.CIFAR10(root="F:\python_and_AI\Hands-On PyTorch",
 train=False, transform=dataset_transform)

#设置数据加载器
test_loader=DataLoader(dataset=test_data,batch_size=4,shuffle=True,
                       num_workers=0,drop_last=False)

#查看dataloader提取的数据
#dataloader会按batch_size对图片进行打包
writer=SummaryWriter("dataloader")
step=0
for data in test_loader:
    imgs, labels = data
    # print(img.shape)    #输出示例：torch.Size([4, 3, 32, 32])，表示4张图片，3通道，32*32
    # print(label)        #输出示例：tensor([6, 0, 0, 9])，表示这四张图片的标签分别为6，0,0,9
    writer.add_images("imgs",imgs,step)
    step += 1
#在tensorboard中查看会发现一个step的图片是一个batch_size的图片拼在一起
writer.close()
