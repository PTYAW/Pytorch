#   损失函数：用来计算实际输出和目标的差距，为我们更新输出提供一定依据（反向传播）
#   Loss Functions

import torch
import torchvision
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Conv2d, MaxPool2d, Flatten, Linear, Sequential
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

for data in dataloader:
    imgs, targets = data
    outputs = xiqiao(imgs)
    result_loss = loss_cross(outputs, targets)
    #   计算梯度，反向传播：方便我们选择优化器
    result_loss.backward()


# inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# targets = torch.tensor([1, 2, 5])
#
# inputs = torch.reshape(inputs, (1, 1, 1, 3))
# targets = torch.reshape(targets, (1, 1, 1, 3))
#
# #   可以直接调用L1Loss损失函数
# #   L1Loss：求平均绝对误差
# loss = L1Loss()
# result = loss(inputs, targets)
#
# #   MSELoss：求均方差
# loss_mse = MSELoss()
# result_mse = loss_mse(inputs, targets)
#
# #   CROSSENTROPYLOSS：交叉熵
# inputs_1 = torch.tensor([0.1, 0.2, 0.3])
# targets_1 = torch.tensor([1])
# inputs_1 = torch.reshape(inputs_1, (1, 3))
# loss_cross = CrossEntropyLoss()
# result_cross = loss_cross(inputs_1, targets_1)
#
# print(result)
# print(result_mse)
# print(result_cross)
