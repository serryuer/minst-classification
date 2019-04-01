# Minst 手写数字识别-Pytorch-AlexNet-ResNet34

本项目是深度学习课程的第一次实验作业，数据集地址：http://yann.lecun.com/exdb/mnist/，本项目主要参考《深度学习框架爱 Pytorch入门与实践》一书中所有使用的代码结构和方法工具。

## 文件结构

- checkpoints：保存训练好的模型
- data：保存数据集以及对数据的加载和预处理程序
- models：保存封装好的模型代码
- utils：工具，这里只有一个可视化的工具，即Visdom的封装
- config.py：项目中常用参数的配置
- main.py：项目入口，包括数据加载、训练、评估、测试
- requirements.txx：项目依赖

## 安装

```
pip -r requirements.txt
```

## 训练

训练前必须先启动visdom
```
python -m visdom.server
```



