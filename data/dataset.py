# coding:utf8
import gzip

import numpy as np
from torch.utils import data

from config import opt


class Minst(data.Dataset):

    def __init__(self, data_root, label_root=None, transforms=None, train=True, test=False):
        """
        获取所有图片，并根据训练、验证、测试划分数据
        :param data_root:数据路径
        :param transforms:数据转换操作
        :param train:是否训练集
        :param test:是否测试集
        """

        self.test = test

        if test:
            self.image_nums = opt.test_image_nums
        else:
            self.image_nums = opt.train_image_nums

        # 从文件中读取图片数据
        with gzip.open(data_root) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(opt.image_size * opt.image_size * self.image_nums * opt.num_channels)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = (data - (opt.pixel_depth / 2.0)) / opt.pixel_depth
            self.data = data.reshape(self.image_nums, opt.image_size, opt.image_size, opt.num_channels)

        # 从文件中读取图片标签数据
        if not test:
            with gzip.open(label_root) as bytestream:
                bytestream.read(8)
                buf = bytestream.read(1 * self.image_nums)
                self.labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        # 划分训练集和验证集
        if test:
            pass
        elif train:
            self.image_nums = opt.train_image_nums * 0.7
            self.data = self.data[:int(opt.train_image_nums * 0.7)]
            self.labels = self.labels[:int(opt.train_image_nums * 0.7)]
        else:
            self.image_nums = opt.train_image_nums - opt.train_image_nums * 0.7
            self.data = self.data[int(opt.train_image_nums * 0.7):]
            self.labels = self.labels[int(opt.train_image_nums * 0.7):]

            # 数据预处理操作

    def __getitem__(self, index):
        """
        返回一张图片的数据，如果是测试集没有label
        :param index:
        :return:
        """
        if self.test:
            return T.ToTensor()(self.data[index]), index
        else:
            return T.ToTensor()(self.data[index]), self.labels[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    train_data = Minst(data_root="train/train-images.gz", label_root="train/train-labels.gz", train=True)
    train_dataloader = data.DataLoader(train_data, opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)
    import torchvision.transforms as T

    for ii, (data, target) in enumerate(train_dataloader):
        toimage = T.ToPILImage()
        image = data[ii].numpy()
        image = (image + 0.5) * 255
        result = toimage(T.ToTensor()(image))
        result.show()
        print(target[ii])
