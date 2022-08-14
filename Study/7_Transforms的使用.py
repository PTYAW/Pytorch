#   transform工具箱：将图片变成我们想要的结果
#   tensor数据类型，transform.ToTensor
#   1、transform该如何被使用（python）
#   2、为什么我们需要totensor数据类型

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
#   生成一个totensor型的对象
tensor_train = transforms.ToTensor()
#   生成tensor类型的照片：一个三位数组记录每个像素点的R、G、B的强度
tensor_img = tensor_train(img)

#   生成日志文件
write = SummaryWriter("logs")
write.add_image("Tensor_img", tensor_img)
write.close()

