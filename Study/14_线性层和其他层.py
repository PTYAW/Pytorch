#   正则化层：Normalization Layers
#   Recurren Layers :用于文字识别
#   Transformer Layers
#   Dropout Layers：随机失活，防止过拟合

#   线性层：Linear Layers   用于变化特征维度

import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("dataset_CIF10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Xiqiao(nn.Module):
    def __init__(self):
        super(Xiqiao, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, shuru):
        shuchu = self.linear1(shuru)
        return shuchu


xiqiao = Xiqiao()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    #   将图片变成一条
    # imgs_output = torch.reshape(imgs, (1, 1, 1, -1))
    imgs_output = torch.flatten(imgs)
    print(imgs_output.shape)
    imgs_output = xiqiao(imgs_output)
    print(imgs_output.shape)
