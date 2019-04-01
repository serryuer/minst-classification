# coding:utf8
import time

import torch as t


class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        可加载指定路径的模型
        :param path: 模型路径
        :return:
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用模型名字+时间作为文件名
        :param name: 模型名字d
        :return:
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + "_"
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
