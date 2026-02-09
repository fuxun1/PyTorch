from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

'''示例1：将图片PIL数据类型转换为tensor类型，并利用tensorboard进行查看'''
img_path="F:\\python_and_AI\\ants_bee_dataset\\train\\ants\\0013035.jpg"
img = Image.open(img_path)  #得到PIL格式的图片
print(type(img))    # <class 'PIL.JpegImagePlugin.JpegImageFile'>
tensor_trans = transforms.ToTensor()    #创建类ToTensor的实例化对象
tensor_img = tensor_trans(img)  # 转换
print(type(tensor_img))     # <class 'torch.Tensor'>

writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
writer.close()

'''示例2：将numpy.ndarray图片类型转化为tensor类型'''
cv_img = cv2.imread(img_path)
tensor_cv_img = tensor_trans(cv_img)
print(type(tensor_cv_img))
