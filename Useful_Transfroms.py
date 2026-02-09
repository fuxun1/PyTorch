#transfroms的一些常用工具
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path="E:\\DataSet\\ants_bee_dataset\\train\\ants\\258217966_d9d90d18d3.jpg"
img = Image.open(img_path)

trans_tensor=transforms.ToTensor()
img_tensor=trans_tensor(img)

writer=SummaryWriter("logs")
writer.add_image("tensor_img",img_tensor)

#Normalize 归一化
print(img_tensor[0,0,0])    #查看未均一化时的第1行第1列第1个通道的像素点的数值(实际上第一个数是通道，tensor为CHW)
img_trans_normalize=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = img_trans_normalize(img_tensor)
print(img_norm[0,0,0])  #查看均一化后的该像素点的数值
writer.add_image("tensor_img_normalize",img_norm)

#Resize 裁剪
print(img.size)
trans_resize=transforms.Resize((512,512))    #实例化对象(工具)并设定参数
img_resize=trans_resize(img)
print(img_resize.size)
img_resize_tensor=trans_tensor(img_resize)     #再将裁剪后的PIL图片转化为tensor类型
writer.add_image("img_resize_tensor",img_resize_tensor)

#compose  组合
trans_resize_2=transforms.Resize((512,512))
trans_compose=transforms.Compose([trans_resize_2,trans_tensor])
img_resize_compose=trans_compose(img)
writer.add_image("img_resize_compose",img_resize_compose)

#RandomCrop 随机裁剪
trans_random=transforms.RandomCrop((300))
trans_compose_2=transforms.Compose([trans_random,trans_tensor])
#随机裁剪10份
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("Random_crop",img_crop,global_step=i)


writer.close()  #关闭日志

