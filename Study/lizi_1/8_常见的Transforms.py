#   这里介绍了transfors常用的一些工具

#   输入  PIL     Image.open()
#   输出  tensor  Totensor()
#   作用  narrays cv.imread()

from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

#   导入图片
img_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
#   生成日志文件
write = SummaryWriter("logs")

#   生成tensor文件
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
#   上传到日志上
write.add_image("Tensor", img_tensor)

#   Normalize：归一化   对图像进行标准化（均值为0，标准差为1）
#   output[channel] = (input[channel] - mean[channel]) / std[channel]
#   参数：mean：各通道的均值  std：各通道的标准差
# [0.5, 0.5, 0.5], [0.5, 0.5, 0.5] 第一次 [9, 3, 3], [9, 8, 1]第二次
trans_norm = transforms.Normalize([9, 3, 3], [9, 8, 1])
img_norm = trans_norm(img_tensor)
write.add_image("Normalize", img_norm, 2)

#   Resize:更改尺寸
#   参数：size：大小（高，宽） 如果只输入一个就是等比缩放
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
#   转换照片格式
img_resize = trans_totensor(img_resize)
write.add_image("Resize", img_resize, 1)

#   Compose - resize - 2 :等比缩放  ,只要改变高，宽就等比进行缩放
#   Compose的参数：[]：一个列表（必须是transiforms类型）
#   Compose的作用：将几个步骤整合在一起
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
write.add_image("Resize_2", img_resize_2, 1)

#   Random：随机裁剪
#   参数：size:一个参数就是正方形，或者输入（高，宽）
trans_random = transforms.RandomCrop((211, 300))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(0, 10):
    img_random = trans_compose_2(img)
    write.add_image("RandomCrop", img_random, i)


write.close()
