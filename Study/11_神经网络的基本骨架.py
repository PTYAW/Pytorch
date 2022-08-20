import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("dataset_CIF10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Xiqiao(nn.Module):
    def __init__(self):
        #   super函数用来简化初始化，子类继承父类
        super(Xiqiao, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, shuru):
        shuchu = self.conv1(shuru)
        return shuchu


xiqiao = Xiqiao()
print(xiqiao)

for data in dataloader:
    imgs, targets = data
    img_shuchu = xiqiao(imgs)
    print(img_shuchu.shape)
