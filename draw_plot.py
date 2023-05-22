import argparse
import csv
from functools import reduce
from tqdm import trange
import importlib

import utils
from plot_utils import line_chart
from plot_utils import visualization
from plot_utils.confusion_matrix import plot_confusion_matrix
from plot_utils.last_layer_analysis import last_layer_analysis, last_layer_analysis_2
from datasets import MyDataset
from networks import MyModel
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders

import initializion
import torch
import torchinfo
import os
import os.path
import numpy as np

"""
绘图工具
版本说明:
V1
时间: 2022.10.22
作者: fff
说明: plot_confusion_matrix, plot_line_chart
V2
时间: 2022.10.27
作者: fff
说明: get_csv_data, plot_multiple_line_chart
V3
时间: 2022.11.12
作者: fff
说明: zone_and_linked局部放大折线图
"""


def get_csv_data(dir_path, filename):
    """
    从csv文件获取数据
    参数说明:
    输入:
        dir_path: 文件夹位置
        filename: 文件名称
    输出:
        data (narray)
    """
    csvfile = open(dir_path + filename, 'r')
    read = csv.reader(csvfile)
    data = [r for r in read]
    data = np.array(data)
    return data


def print_acc():
    # 打印测试准确率折线图
    dir_path = './result/lstm/data/'

    version = 0
    filename = version + 'lstm_acc.csv'
    acc = get_csv_data(dir_path=dir_path, filename=filename)
    snr = acc[1:, 0].astype(np.int)  # 从string转化为int
    # print(snr)
    xlabel_name = acc[0, 0]
    test_acc = acc[1:, 1].astype(np.float)
    ylabel_name = acc[0, 1]
    dir_path = './result/lstm/data/'
    label = 'test_acc'
    title = version + 'lstm classification test acc'
    line_chart.plot_line_chart(snr, test_acc, dir_path=dir_path, label=label, title=title, xlabel_name=xlabel_name,
                               ylabel_name=ylabel_name)


def print_loss():
    # 打印训练损失值折线图
    dir_path = './result/lstm/data/'
    filename = 'lstm_loss.csv'
    loss = get_csv_data(dir_path=dir_path, filename=filename)
    epoch = loss[1:, 0].astype(np.int)  # 从string转化为int
    xlabel_name = loss[0, 0]
    train_loss = loss[1:, 1].astype(np.float)
    ylabel_name = loss[0, 1]
    dir_path = './result/lstm/data/'
    label = 'train_loss'
    title = 'lstm classification train loss'
    line_chart.plot_line_chart(epoch, train_loss, dir_path=dir_path, label=label, title=title, xlabel_name=xlabel_name,
                               ylabel_name=ylabel_name)


def print_compare_acc():
    dir_path = './result/lstm/data/'
    filename1 = 'lstm_acc.csv'
    filename2 = 'PolyLoss_lstm_acc.csv'
    acc1 = get_csv_data(dir_path=dir_path, filename=filename1)
    acc2 = get_csv_data(dir_path=dir_path, filename=filename2)

    test_acc1 = acc1[1:, 1].astype(np.float)
    test_acc2 = acc2[1:, 1].astype(np.float)

    snr1 = acc1[1:, 0].astype(np.int)  # 从string转化为int
    snr2 = acc2[1:, 0].astype(np.int)  # 从string转化为int
    snr = [snr1, snr2]
    xlabel_name = acc1[0, 0]
    ylabel_name = acc1[0, 1]
    # 水平拼接
    # test_acc1 = test_acc1.reshape(-1, 1)
    # test_acc2 = test_acc2.reshape(-1, 1)
    # test_acc = np.hstack((test_acc1, test_acc2))
    test_acc = [test_acc1, test_acc2]

    dir_path = './result/lstm/data/'
    label = ['CrossEntropyLoss', 'PolyLoss']
    title = 'cross_poly_loss acc'
    color = ['red', 'blue']
    line_chart.plot_multiple_line_chart(snr, test_acc, dir_path=dir_path, color=color, label=label, title=title,
                                        xlabel_name=xlabel_name, ylabel_name=ylabel_name, flag=False)


def print_compare_train_loss():
    dir_path = './result/lstm/data/'
    filename1 = 'lstm_loss.csv'
    filename2 = 'PolyLoss_lstm_loss.csv'
    loss1 = get_csv_data(dir_path=dir_path, filename=filename1)
    loss2 = get_csv_data(dir_path=dir_path, filename=filename2)

    train_loss1 = loss1[1:, 1].astype(np.float)
    train_loss2 = loss2[1:, 1].astype(np.float)

    epochs1 = loss1[1:, 0].astype(np.int)  # 从string转化为int
    epochs2 = loss2[1:, 0].astype(np.int)  # 从string转化为int
    epochs = [epochs1, epochs2]
    xlabel_name = loss1[0, 0]
    ylabel_name = loss1[0, 1]
    # 水平拼接
    # train_loss1 = train_loss1.reshape(-1, 1)
    # train_loss2 = train_loss2.reshape(-1, 1)
    # train_loss = np.hstack((train_loss1, train_loss2))
    train_loss = [train_loss1, train_loss2]

    dir_path = './result/lstm/data/'
    label = ['CrossEntropyLoss', 'PolyLoss']
    title = 'cross_poly_loss loss'
    color = ['red', 'blue']
    line_chart.plot_multiple_line_chart(epochs, train_loss, dir_path=dir_path, color=color, label=label, title=title,
                                        xlabel_name=xlabel_name, ylabel_name=ylabel_name)


def print_train_val_loss():
    dir_path = './result/lstm/data/'
    filename1 = 'PolyLoss_lstm_loss.csv'
    # filename1 = 'lstm_loss.csv'
    # filename1 = 'PETCGDNN_lstm_loss.csv'

    loss1 = get_csv_data(dir_path=dir_path, filename=filename1)

    train_loss1 = loss1[1:, 1].astype(np.float)
    train_loss2 = loss1[1:, 2].astype(np.float)

    epochs1 = loss1[1:, 0].astype(np.int)  # 从string转化为int
    epochs2 = loss1[1:, 0].astype(np.int)  # 从string转化为int
    epochs = [epochs1, epochs2]
    xlabel_name = loss1[0, 0]
    ylabel_name = loss1[0, 1] + '_' + loss1[0, 2]
    # 水平拼接
    # train_loss1 = train_loss1.reshape(-1, 1)
    # train_loss2 = train_loss2.reshape(-1, 1)
    # train_loss = np.hstack((train_loss1, train_loss2))
    train_loss = [train_loss1, train_loss2]

    dir_path = './result/lstm/data/'
    label = [loss1[0, 1], loss1[0, 2]]
    title = 'PolyLoss_lstm_train_val loss'
    color = ['red', 'blue']
    line_chart.plot_multiple_line_chart(epochs, train_loss, dir_path=dir_path, color=color, label=label, title=title,
                                        xlabel_name=xlabel_name, ylabel_name=ylabel_name, flag=False)


def normalize_data(X):
    min_value = X.min()
    max_value = X.max()

    X = (X - min_value) / (max_value - min_value)
    X = np.float32(X)
    return X


def get_test_query_dataset(num, transform_x=None, transform_y=None):
    """
    用于可视化提取数据
    :param num:
    :param transform_x:
    :param transform_y:
    :return:
    """
    x = np.load('../SplitFSSEIDataset/X_test_{}Class.npy'.format(num))
    y = np.load('../SplitFSSEIDataset/Y_test_{}Class.npy'.format(num))
    if transform_x:
        x = transform_x(x)
    if transform_y:
        y = transform_y(y)

    y = y.astype(np.uint8)

    return x, y


def print_visualization():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args, extra_args = initializion.init()
    args.results_path = os.path.expanduser(args.results_path)  # 把 path 中包含的 ~ 和 ~user 转换成用户目录
    # incremental_learning 类中基本参数
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    utils.seed_everything(seed=args.seed)
    # False:固定cuda的随机数种子, 在seed_everything 中utils.cudnn_deterministic已经设定为True
    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' +
              arg + ':', getattr(args, arg))
    print('=' * 108)
    print('Extra Arguments =')
    print(extra_args)
    print('=' * 108)

    # cuda args
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print("WARNING: [CUDA unavailable] Using CPU instead!")
        device = 'cpu'
    # networks args
    from networks.network import LLL_Net
    net = getattr(importlib.import_module(name='networks'), args.network)
    init_model = net(pretrained=False)

    # approach args
    from approachs import incremental_learning
    Appr = getattr(importlib.import_module(name='approachs.' + args.approach), 'Appr')
    assert issubclass(Appr, incremental_learning.Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)

    print('=' * 108)
    print('Approach Arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' +
              arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # examplars args
    from datasets import exemplars_dataset
    Appr_examplars_dataset = Appr.exemplars_dataset_class()
    if Appr_examplars_dataset:
        assert issubclass(Appr_examplars_dataset, exemplars_dataset.ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_examplars_dataset.extra_parser(extra_args)
        print('=' * 108)
        print('Examplars Arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' +
                  arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    # logger args
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    # print(full_exp_name)
    full_exp_name = full_exp_name + '_' + args.approach + '_' + args.network
    if args.exp_name is not None:
        full_exp_name = full_exp_name + '_' + args.exp_name

    logger = MultiLogger(args.results_path, full_exp_name, num_tasks=args.num_tasks, loggers=args.log,
                         save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(datasets=args.datasets,
                                                              num_tasks=args.num_tasks,
                                                              nc_first_task=args.nc_first_task,
                                                              batch_size=args.batch_size,
                                                              num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)

    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Appr, examplar_dataset instance
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    net.schedule_step = args.schedule_step
    # add heads
    for t, ncls in taskcla:
        net.add_head(ncls)

    model_load_path = os.path.join(args.results_path, full_exp_name, 'models', '{}tasks'.format(args.num_tasks),
                                   'task{}.ckpt'.format(args.num_tasks - 1))
    print("model load from: ", model_load_path)
    net.load_state_dict(torch.load(model_load_path))
    net.to(device)

    utils.seed_everything(seed=args.seed)
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform_x, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_examplars_dataset:
        appr_kwargs['exemplars_dataset'] = Appr_examplars_dataset(transform=None,
                                                                  class_indices=class_indices,
                                                                  **appr_exemplars_dataset_args.__dict__)
    appr = Appr(net, device, **appr_kwargs)
    all_embs = []
    all_data = []
    all_targets = []
    all_targets_for_task = []
    net.eval()
    print(taskcla)
    start_label = 0
    dir_path = os.path.join(args.results_path, full_exp_name,
                            'figures', '{}tasks'.format(args.num_tasks))
    os.makedirs(dir_path, exist_ok=True)
    for t in range(args.num_tasks):
        # print(len(tst_loader[t].dataset))
        '''
        loss, acc_taw, acc_tag = appr.eval(t, tst_loader[t])
        print('loss: ', loss)
        print('acc_taw: ', acc_taw)
        print('acc_tag: ', acc_tag)
        '''
        task_data = []
        task_embs = []
        task_targets = []
        for batch, (images, targets) in enumerate(tst_loader[t]):
            with torch.no_grad():
                embs = net.model(images.to(device))
            task_data.append(images.cpu())
            task_embs.append(embs.cpu())
            task_targets.append(targets)
        all_data.append(torch.cat(task_data, dim=0))
        all_embs.append(torch.cat(task_embs, dim=0))
        all_targets.append(torch.cat(task_targets, dim=0))
        all_targets_for_task.append((torch.zeros_like(torch.cat(task_targets, dim=0)) + t))

        emb_path = os.path.join(dir_path, 'emb_task{}.png'.format(t))
        labels = range(90)
        visualization.plot_visual(torch.cat(all_embs, dim=0), torch.cat(all_targets, dim=0), emb_path, labels)
        # 单个任务
        emb_path = os.path.join(dir_path, 'only_emb_task{}.png'.format(t))
        labels = range(start_label, start_label + taskcla[t][1])
        start_label += taskcla[t][1]
        visualization.plot_visual(torch.cat(task_embs, dim=0), torch.cat(task_targets, dim=0), emb_path, labels)

        # data_path = os.path.join(args.results_path, full_exp_name, 'figures', 'data_task{}.png'.format(t))
        # all_data = torch.cat(all_data, dim=0)
        # all_data = torch.reshape(all_data, (all_data.size(0), -1))
        # print(all_data.shape)
        # visualization.plot_visual(all_data, torch.cat(all_targets, dim=0), data_path)

    emb_path = os.path.join(dir_path, 'emb_task.png')
    labels = [i for i in range(args.num_tasks)]
    visualization.plot_visual(torch.cat(all_embs, dim=0), torch.cat(all_targets_for_task, dim=0), emb_path, labels)

    '''
    X_new = torch.zeros(len(tst_loader), 1024))
    for t in range(args.num_tasks):
        embs = net.model()
    with torch.no_grad():
        for x_idx in trange(X_test.size(0)):
            # print(X_test[x_idx, ...].unsqueeze(0).shape)
            X_new[x_idx, :], _ = encoder(X_test[x_idx, ...].unsqueeze(0))

        X_new = X_new.cpu().numpy()
        visualization.plot_visual(X_new, Y_test, whole_dir_path)
    '''


def print_trainset_distribution():
    # 获取训练集的每类样本数量的分布
    args = initializion.init()
    num = args.train_val_dataset_num
    filepath_y = '../SplitFSSEIDataset/Y_split_train_{}Classes.npy'.format(num)
    y = np.load(filepath_y)
    y = y.astype(np.uint8)
    classes = np.unique(y)
    num_per_class = []
    for c in classes:
        num_per_class.append((c == y).sum())
    num_per_class = np.array(num_per_class)
    dir_path = args.data_dir_path + './dataset'
    os.makedirs(dir_path, exist_ok=True)

    whole_path = dir_path + '/'
    title = 'splited_trainset_distribution'
    label = 'trainset'
    xlabel_name = 'class'
    ylabel_name = 'num'
    line_chart.plot_line_chart(classes, num_per_class, whole_path, label=label, title=title,
                               xlabel_name=xlabel_name, ylabel_name=ylabel_name)
    # print(num_per_class.sum())
    # print(y.shape)

    sorted_num_per_class_idx = np.argsort(num_per_class)  # 获取升序后的下标
    print(sorted_num_per_class_idx)
    sorted_num_per_class = num_per_class[sorted_num_per_class_idx[::-1]]  # 降序
    title = 'splited_sorted_trainset_distribution'
    label = 'sorted_trainset'
    xlabel_name = 'class'
    ylabel_name = 'num'

    line_chart.plot_line_chart(classes, sorted_num_per_class, whole_path, label=label, title=title,
                               xlabel_name=xlabel_name, ylabel_name=ylabel_name)


def print_confusion_matrix():
    args, extra_args = initializion.init()
    args.results_path = os.path.expanduser(args.results_path)  # 把 path 中包含的 ~ 和 ~user 转换成用户目录
    # incremental_learning 类中基本参数
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    utils.seed_everything(seed=args.seed)
    # False:固定cuda的随机数种子, 在seed_everything 中utils.cudnn_deterministic已经设定为True
    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' +
              arg + ':', getattr(args, arg))
    print('=' * 108)
    print('Extra Arguments =')
    print(extra_args)
    print('=' * 108)

    # cuda args
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print("WARNING: [CUDA unavailable] Using CPU instead!")
        device = 'cpu'
    # networks args
    from networks.network import LLL_Net
    net = getattr(importlib.import_module(name='networks'), args.network)
    init_model = net(pretrained=False)

    # approach args
    from approachs import incremental_learning
    Appr = getattr(importlib.import_module(name='approachs.' + args.approach), 'Appr')
    assert issubclass(Appr, incremental_learning.Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)

    print('=' * 108)
    print('Approach Arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' +
              arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # examplars args
    from datasets import exemplars_dataset
    Appr_examplars_dataset = Appr.exemplars_dataset_class()
    if Appr_examplars_dataset:
        assert issubclass(Appr_examplars_dataset, exemplars_dataset.ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_examplars_dataset.extra_parser(extra_args)
        print('=' * 108)
        print('Examplars Arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' +
                  arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    # logger args
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    # print(full_exp_name)
    full_exp_name = full_exp_name + '_' + args.approach + '_' + args.network
    if args.exp_name is not None:
        full_exp_name = full_exp_name + '_' + args.exp_name

    logger = MultiLogger(args.results_path, full_exp_name, num_tasks=args.num_tasks, loggers=args.log,
                         save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(datasets=args.datasets,
                                                              num_tasks=args.num_tasks,
                                                              nc_first_task=args.nc_first_task,
                                                              batch_size=args.batch_size,
                                                              num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)

    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Appr, examplar_dataset instance
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    net.schedule_step = args.schedule_step
    # add heads
    for t, ncls in taskcla:
        net.add_head(ncls)
    model_load_path = os.path.join(args.results_path, full_exp_name, 'models', '{}tasks'.format(args.num_tasks),
                                   'task{}.ckpt'.format(args.num_tasks - 1))
    print("model load from: ", model_load_path)
    net.load_state_dict(torch.load(model_load_path))
    net.to(device)
    utils.seed_everything(seed=args.seed)
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform_x, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_examplars_dataset:
        appr_kwargs['exemplars_dataset'] = Appr_examplars_dataset(transform=None,
                                                                  class_indices=class_indices,
                                                                  **appr_exemplars_dataset_args.__dict__)
    appr = Appr(net, device, **appr_kwargs)
    conf_taw = np.zeros((len(class_indices), len(class_indices)))
    conf_tag = np.zeros((len(class_indices), len(class_indices)))

    cm_path = os.path.join(args.results_path, full_exp_name,
                            'figures', '{}tasks'.format(args.num_tasks))
    os.makedirs(cm_path, exist_ok=True)
    for t in range(args.num_tasks):
        with torch.no_grad():
            net.eval()
            for batch, (images, targets) in enumerate(tst_loader[t]):
                outputs = net(images.to(device))
                pred_taw = torch.zeros_like(targets.to(device))
                # Task-Aware Multi-Head
                for m in range(len(targets)):
                    this_task = (net.task_cls.cumsum(0) <= targets[m]).sum()
                    pred_taw[m] = outputs[this_task][m].argmax() + net.task_offset[this_task]
                pred_tag = torch.cat(outputs, dim=1).argmax(1)
                # print(pred.shape)
                for i in range(0, pred_taw.shape[0]):
                    # 行为正确模式， 列为预测模式
                    j = targets[i]
                    k = pred_taw[i]
                    conf_taw[j, k] = conf_taw[j, k] + 1
                for i in range(0, pred_tag.shape[0]):
                    # 行为正确模式， 列为预测模式
                    j = targets[i]
                    k = pred_tag[i]
                    conf_tag[j, k] = conf_tag[j, k] + 1

    plot_confusion_matrix(conf_taw, cm_path, title='cm_taw', labels=range(len(class_indices)), normalize=True)
    plot_confusion_matrix(conf_tag, cm_path, title='cm_tag', labels=range(len(class_indices)), normalize=True)


def print_last_layer_params():
    """比较weights, bias L2范数大小"""
    args, extra_args = initializion.init()
    args.results_path = os.path.expanduser(args.results_path)  # 把 path 中包含的 ~ 和 ~user 转换成用户目录
    # incremental_learning 类中基本参数
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    utils.seed_everything(seed=args.seed)
    # False:固定cuda的随机数种子, 在seed_everything 中utils.cudnn_deterministic已经设定为True
    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' +
              arg + ':', getattr(args, arg))
    print('=' * 108)
    print('Extra Arguments =')
    print(extra_args)
    print('=' * 108)

    # cuda args
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print("WARNING: [CUDA unavailable] Using CPU instead!")
        device = 'cpu'
    # networks args
    from networks.network import LLL_Net
    net = getattr(importlib.import_module(name='networks'), args.network)
    init_model = net(pretrained=False)

    # approach args
    from approachs import incremental_learning
    Appr = getattr(importlib.import_module(name='approachs.' + args.approach), 'Appr')
    assert issubclass(Appr, incremental_learning.Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)

    print('=' * 108)
    print('Approach Arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' +
              arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # examplars args
    from datasets import exemplars_dataset
    Appr_examplars_dataset = Appr.exemplars_dataset_class()
    if Appr_examplars_dataset:
        assert issubclass(Appr_examplars_dataset, exemplars_dataset.ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_examplars_dataset.extra_parser(extra_args)
        print('=' * 108)
        print('Examplars Arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' +
                  arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    # logger args
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    # print(full_exp_name)
    full_exp_name = full_exp_name + '_' + args.approach + '_' + args.network
    if args.exp_name is not None:
        full_exp_name = full_exp_name + '_' + args.exp_name

    logger = MultiLogger(args.results_path, full_exp_name, num_tasks=args.num_tasks, loggers=args.log,
                         save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(datasets=args.datasets,
                                                              num_tasks=args.num_tasks,
                                                              nc_first_task=args.nc_first_task,
                                                              batch_size=args.batch_size,
                                                              num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)

    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Appr, examplar_dataset instance
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    net.schedule_step = args.schedule_step
    # add heads
    for t, ncls in taskcla:
        net.add_head(ncls)
        model_load_path = os.path.join(args.results_path, full_exp_name, 'models', '{}tasks'.format(args.num_tasks),
                                       'task{}.ckpt'.format(t))
        print("model load from: ", model_load_path)
        net.load_state_dict(torch.load(model_load_path))
        net.to(device)
        utils.seed_everything(seed=args.seed)
        first_train_ds = trn_loader[0].dataset
        transform, class_indices = first_train_ds.transform_x, first_train_ds.class_indices
        appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
        if Appr_examplars_dataset:
            appr_kwargs['exemplars_dataset'] = Appr_examplars_dataset(transform=None,
                                                                      class_indices=class_indices,
                                                                      **appr_exemplars_dataset_args.__dict__)
        appr = Appr(net, device, **appr_kwargs)

        name = 'weights_{}tasks'.format(t)
        weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=False)
        logger.log_figure(name=name, iter=0, num_tasks=args.num_tasks, figure=weights)
        name = 'bias_{}tasks'.format(t)
        logger.log_figure(name=name, iter=0, num_tasks=args.num_tasks, figure=biases)

        name = 'weights_sorted_{}tasks'.format(t)
        weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
        logger.log_figure(name=name, iter=0, num_tasks=args.num_tasks, figure=weights)
        name = 'bias_sorted_{}tasks'.format(t)
        logger.log_figure(name=name, iter=0, num_tasks=args.num_tasks, figure=biases)


def print_last_layer_params_2():
    """比较类特征与对应分类器的正负号一致性"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args, extra_args = initializion.init()
    args.results_path = os.path.expanduser(args.results_path)  # 把 path 中包含的 ~ 和 ~user 转换成用户目录
    # incremental_learning 类中基本参数
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    utils.seed_everything(seed=args.seed)
    # False:固定cuda的随机数种子, 在seed_everything 中utils.cudnn_deterministic已经设定为True
    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' +
              arg + ':', getattr(args, arg))
    print('=' * 108)
    print('Extra Arguments =')
    print(extra_args)
    print('=' * 108)

    # cuda args
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print("WARNING: [CUDA unavailable] Using CPU instead!")
        device = 'cpu'
    # networks args
    from networks.network import LLL_Net
    net = getattr(importlib.import_module(name='networks'), args.network)
    init_model = net(pretrained=False)

    # approach args
    from approachs import incremental_learning
    Appr = getattr(importlib.import_module(name='approachs.' + args.approach), 'Appr')
    assert issubclass(Appr, incremental_learning.Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)

    print('=' * 108)
    print('Approach Arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' +
              arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # examplars args
    from datasets import exemplars_dataset
    Appr_examplars_dataset = Appr.exemplars_dataset_class()
    if Appr_examplars_dataset:
        assert issubclass(Appr_examplars_dataset, exemplars_dataset.ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_examplars_dataset.extra_parser(extra_args)
        print('=' * 108)
        print('Examplars Arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' +
                  arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    # logger args
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    # print(full_exp_name)
    full_exp_name = full_exp_name + '_' + args.approach + '_' + args.network
    if args.exp_name is not None:
        full_exp_name = full_exp_name + '_' + args.exp_name

    logger = MultiLogger(args.results_path, full_exp_name, num_tasks=args.num_tasks, loggers=args.log,
                         save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(datasets=args.datasets,
                                                              num_tasks=args.num_tasks,
                                                              nc_first_task=args.nc_first_task,
                                                              batch_size=args.batch_size,
                                                              num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)

    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Appr, examplar_dataset instance
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    net.schedule_step = args.schedule_step
    # add heads
    for t, ncls in taskcla:
        net.add_head(ncls)
        model_load_path = os.path.join(args.results_path, full_exp_name, 'models', '{}tasks'.format(args.num_tasks),
                                       'task{}.ckpt'.format(t))
        print("model load from: ", model_load_path)
        net.load_state_dict(torch.load(model_load_path))
        net.to(device)
        utils.seed_everything(seed=args.seed)
        first_train_ds = trn_loader[0].dataset
        transform, class_indices = first_train_ds.transform_x, first_train_ds.class_indices
        appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
        if Appr_examplars_dataset:
            appr_kwargs['exemplars_dataset'] = Appr_examplars_dataset(transform=None,
                                                                      class_indices=class_indices,
                                                                      **appr_exemplars_dataset_args.__dict__)
        appr = Appr(net, device, **appr_kwargs)

        all_embs = []
        all_targets = []
        net.eval()
        print(taskcla)
        dir_path = os.path.join(args.results_path, full_exp_name,
                                'figures', '{}tasks'.format(args.num_tasks))
        os.makedirs(dir_path, exist_ok=True)
        # 获取所有任务对应的类的特征映射
        for tt in range(t+1):
            # print(len(tst_loader[t].dataset))
            '''
            loss, acc_taw, acc_tag = appr.eval(t, tst_loader[t])
            print('loss: ', loss)
            print('acc_taw: ', acc_taw)
            print('acc_tag: ', acc_tag)
            '''
            task_data = []
            task_embs = []
            task_targets = []
            for batch, (images, targets) in enumerate(tst_loader[tt]):
                with torch.no_grad():
                    embs = net.model(images.to(device))
                task_data.append(images.cpu())
                task_embs.append(embs.cpu())
                task_targets.append(targets)
            all_embs.append(torch.cat(task_embs, dim=0))
            all_targets.append(torch.cat(task_targets, dim=0))

        name = 'consistency_{}tasks'.format(t)
        weights = last_layer_analysis_2(net.heads, all_embs, all_targets, t,
                                        taskcla, y_lim=True, sort_weights=False)
        logger.log_figure(name=name, iter=0, num_tasks=args.num_tasks, figure=weights)

        name = 'consistency_sorted_{}tasks'.format(t)
        weights = last_layer_analysis_2(net.heads, all_embs, all_targets, t,
                                        taskcla, y_lim=True, sort_weights=True)
        logger.log_figure(name=name, iter=0, num_tasks=args.num_tasks, figure=weights)



if __name__ == '__main__':
    # print_loss()
    # print_acc()
    # print_compare_train_loss()
    # print_compare_acc()
    # print_train_val_loss()
    # print_acc()
    print_visualization()
    print_confusion_matrix()
    print_last_layer_params()
    print_last_layer_params_2()
    # print_trainset_distribution()
