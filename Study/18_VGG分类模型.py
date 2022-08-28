#   torchvision.models
#   使用的数据集：ImageNet

import torchvision
from torch import nn

#   pretrained=True,返回一个在ImageNet中训练好的模型
#   pretrained=False,仅返回一些信息
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print("vgg16_true")

train_data = torchvision.datasets.CIFAR10("datset_CIF10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

#   vgg16_true输出层1000，但是CIFAR10输入只有10层
#   1、vgg16_true直接加一层
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
#   2、直接修改最后一层的结构
vgg16_false.classifier[6] = nn.Linear(4096, 10)
