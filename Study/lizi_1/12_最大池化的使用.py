#   nn.MaxPool2d
#   最大池化：利用池化核逐步（stride：步长）覆盖输入图像，取池化核中最大值  (和卷积不同的是，池化核扫过的地方不再扫了)
#   cell_mode为True时，池化核会覆盖到不满足池化核大小(kernel_size)的地方

#   目的：保留图像的特征，并减小数据量       使其训练的更快

import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset_CIF10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)
shuru = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 0, 1, 1],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

#   因为使用池化的函数输入必须是四维（N，C，H，W）
shuru = torch.reshape(shuru, (-1, 1, 5, 5))
print(shuru)


class Xiqiao(nn.Module):
    def __init__(self):
        super(Xiqiao, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input_x):
        output_x = self.maxpool1(input_x)
        return output_x


xiqiao = Xiqiao()
shuchu = xiqiao(shuru)
print(shuchu)

write = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    write.add_images("input", imgs, step)
    imgs_output = xiqiao(imgs)
    write.add_images("output", imgs_output, step)
    step = step+1

write.close()
