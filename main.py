# coding:utf8

import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

import models
from config import opt
from data.dataset import Minst
from utils.visualize import Visualizer


def train(**kwargs):
    """
    训练
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # data
    train_data = Minst(data_root=opt.train_image_path, label_root=opt.train_label_path, train=True)
    val_data = Minst(data_root=opt.train_image_path, label_root=opt.train_label_path, train=False)

    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)

    # 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=opt.weight_decay)

    # 统计指标，平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(10)
    previous_loss = 1e100

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, target) in enumerate(train_dataloader):
            if opt.use_gpu:
                data = data.cuda()
                target = target.cuda()

            # 优化器重置导数
            optimizer.zero_grad()
            # 模型计算
            score = model(data)
            # 根据结果计算损失
            loss = criterion(score, target)
            # 损失回传
            loss.backward()
            # 更新参数
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

        model.save()

        # 计算验证集上的指标以及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch,
            loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()),
            lr=lr
        ))

        # 如果损失不再下降，降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = lr
        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息，用于辅助训练
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(10)
    for ii, (input, target) in tqdm(enumerate(dataloader)):
        if opt.use_gpu:
            input = input.cuda()
            target = target.cuda()
        score = model(input)
        confusion_matrix.add(score.detach().squeeze(), target.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    # TODO fix
    accuracy = 100. * (cm_value.diagonal().sum()) / (cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    """
    测试
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # data
    train_data = Minst(data_root=opt.test_image_path, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (input, path) in tqdm(enumerate(test_dataloader)):
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    # write_csv(results, opt.result_file)

    return results


def help():
    """
    打印辅助信息
    :return: 
    """
    pass


if __name__ == "__main__":
    # fire.Fire()
    train()
