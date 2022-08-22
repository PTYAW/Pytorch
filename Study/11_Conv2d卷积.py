#   nn.Conv2d

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset_CIF10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Xiqiao(nn.Module):
    def __init__(self):
        #   super函数用来简化初始化，子类继承父类
        super(Xiqiao, self).__init__()
        #   输入通道：3，彩色图片RGB3层，输出通道：6，会随机生成6个卷积核对图片进行卷积
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, shuru):
        shuchu = self.conv1(shuru)
        return shuchu


xiqiao = Xiqiao()
print(xiqiao)

write = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    img_shuchu = xiqiao(imgs)
    print(img_shuchu.shape)
    img_shuchu = torch.reshape(img_shuchu, (-1, 3, 30, 30))
    write.add_images("shuru", imgs, step)
    write.add_images("shuchu", img_shuchu, step)
    step = step+1

write.close()
