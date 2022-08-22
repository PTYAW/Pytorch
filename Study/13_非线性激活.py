#   Non-linear Activations：非线性激活
#   nn.ReLU：（瑞鲁）max(0,x)
#   nn.Sigmoid: σ(x)= 1/(1+exp(−x))
#   目的：引入非线性特征（有利于训练出符合特定曲线的模型）

import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset_CIF10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, 64)
shuru = torch.tensor([[1, -0.5],
                      [-1, 3]])

shuru = torch.reshape(shuru, (-1, 1, 2, 2))


class Xiqiao(nn.Module):
    def __init__(self):
        super(Xiqiao, self).__init__()
        #   inplace:True时，无返回值，False时返回一个值。 决定是否替换参数的值
        self.relu1 = ReLU(inplace=False)
        self.sigmoid1 = Sigmoid()

    def forward(self, input_x):
        # output_x = self.relu1(input_x)
        output_x = self.sigmoid1(input_x)
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
