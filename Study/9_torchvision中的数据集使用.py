import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#   参数：root设置根目录,train:是否从训练集创建数据集,transform:数据集的数据类型,download:是否下载数据集，不会重复下载
train_set = torchvision.datasets.CIFAR10(root="./dataset_CIF10", train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root="./dataset_CIF10", train=False, transform=dataset_transform, download=False)

# img, target = test_set[0]
# print(img)
# print(test_set.classes[target])

write = SummaryWriter("../logs")

for i in range(10):
    img, target = test_set[i]
    write.add_image("test_set", img, i)


write.close()
