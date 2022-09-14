#   torch.optim
import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss, Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader


class Xiqiao(nn.Module):
    def __init__(self):
        super(Xiqiao, self).__init__()
        #   一个序列，相当于把所有操作整合了
        self.modle1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.modle1(x)
        return x


xiqiao = Xiqiao()
loss_cross = CrossEntropyLoss()
dataset = torchvision.datasets.CIFAR10("dataset_CIF10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, 64)
#   设置优化器，参数：学习对象的parames，lr：速率
optim = torch.optim.SGD(xiqiao.parameters(), lr=0.01)

for echo in range(0, 20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = xiqiao(imgs)
        result_loss = loss_cross(outputs, targets)
        #   将上个循环的梯度参数清零
        optim.zero_grad()
        #   计算梯度，反向传播：方便我们选择优化器
        result_loss.backward()
        #   对每个参数进行调优
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
