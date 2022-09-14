#   torchvision.models
#   使用的数据集：ImageNet
import torch
import torchvision
from torch import nn

#   pretrained=True,返回一个在ImageNet中训练好的模型
#   pretrained=False,返回一个未经训练的模型
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print("vgg16_true")

train_data = torchvision.datasets.CIFAR10("dataset_CIF10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

#   vgg16_true输出层1000，但是CIFAR10输入只有10层
#   1、vgg16_true直接加一层
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
#   2、直接修改最后一层的结构
vgg16_false.classifier[6] = nn.Linear(4096, 10)

#   模型的保存
vgg16 = torchvision.models.vgg16(pretrained=False)
#   保存方式1:保存了参数和结构
torch.save(vgg16, "vgg16_method1.pth")
#   方式1的读取
# model = torch.load("vgg16_method1.pth")
# print(model)

#   保存方式2:只保存参数(官方推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
#   方式2的读取
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
