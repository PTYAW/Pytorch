#   数据加载
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#   准备测试数据集
test_data = torchvision.datasets.CIFAR10("dataset_CIF10", train=False, transform=torchvision.transforms.ToTensor())
#   参数：要导入的数据集名称，batch_size：每一批的样本数，shuffle：是否每次都洗牌，drop_last：是否删除数据集剩下的项
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(test_data.classes[target])

write = SummaryWriter("logs")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        write.add_images("epoch:{}".format(epoch), imgs, step)
        step = step + 1


write.close()
