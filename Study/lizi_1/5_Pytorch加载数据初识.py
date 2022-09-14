#   Dateset:提供数据去获取数据及label
#   Date loader:为网络提供不同的数据形式
#   实现功能：读取数据

from torch.utils.data import Dataset
from PIL import Image
import os  # 处理文件和目录的模块


# import cv2

#   继承Dataset的类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):  # 构造函数，初始化类 root_dir:根目录地址 label_dil:目标文件夹名称
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)  # 合成目标文件夹路径
        self.img_path_list = os.listdir(self.path)  # 返回一个包含目标路径下所有文件名的列表

    def __getitem__(self, idx):  # 取出数据集中的索引index
        img_name = self.img_path_list[idx]
        img_item_path = os.path.join(self.path, img_name)  # 得到图片的路径
        img = Image.open(img_item_path)  # img存储了图片的信息
        label = self.label_dir  # 赋值label
        return img, label

    def __len__(self):  # 得到长度
        return len(self.img_path_list)


#   声明变量
root_dir = "dataset/hymenoptera_data/train"      # 根目录
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)  # 蚂蚁数据集
bees_dataset = MyData(root_dir, bees_label_dir)  # 蜜蜂数据集
train_dataset = ants_dataset + bees_dataset

img, label = ants_dataset.__getitem__(0)
img.show()
