#   一个可视化工具：记录训练的数据
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

img_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
#   将图片转换成numpy型，也可以cv2.imread（img_PIL）
img_np_array = np.array(img_PIL)

#   生成日志文件
write = SummaryWriter("logs")
#   添加图片到日志文件，参数：标题，图片（要转换格式），序号，数据格式（可以通过img.shape()查看）
write.add_image("test", img_np_array, 1, dataformats='HWC')

for i in range(0,100):
    # 添加一些标量数据到日志中，参数：标题，y轴，x轴
    write.add_scalar("y=2x", i*2, i)


write.close()

#   在终端查看日志文件   =右边接路径
#   tensorboard --logdir=logs


