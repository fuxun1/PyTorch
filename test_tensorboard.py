from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
'''writer是SummaryWriter类的实例对象，用来把训练过程中产生的标量、图片、分布等信息写入日志文件，
供TensorBoard可视化'''
writer = SummaryWriter("logs")

img_path="E:\\DataSet\\ants_bee_dataset\\train\\bees\\16838648_415acd9e3f.jpg"
img = Image.open(img_path)
img_array = np.array(img)
writer.add_image("test",img_array,2,dataformats="HWC")
print(img_array.shape)
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()
