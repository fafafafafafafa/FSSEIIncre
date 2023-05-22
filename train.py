import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import utils
import torch.utils.data
from functools import reduce
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders
import torchinfo
from datasets import MyDataset
from loss_funcs import center_losses
from loss_funcs import triplet_losses
import pytorchtools

import os
import time
import csv
import initializion


def normalize_data(X):
    min_value = X.min()
    max_value = X.max()
    X = (X - min_value) / (max_value - min_value)
    X = np.float32(X)
    return X


def train():

    # 初始化
    args, extra_args = initializion.init()
    args.results_path = os.path.expanduser(args.results_path)   # 把 path 中包含的 ~ 和 ~user 转换成用户目录
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
    Appr = getattr(importlib.import_module(name='approachs.'+args.approach), 'Appr')
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

    logger = MultiLogger(args.results_path, full_exp_name, num_tasks=args.num_tasks, loggers=args.log, save_models=args.save_models)
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
    utils.seed_everything(seed=args.seed)
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform_x, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_examplars_dataset:
        appr_kwargs['exemplars_dataset'] = Appr_examplars_dataset(transform=None,
                                                                  class_indices=class_indices,
                                                                  **appr_exemplars_dataset_args.__dict__)
    appr = Appr(net, device, **appr_kwargs)

    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    acc_tag_topk = np.zeros((max_task, max_task))

    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    forg_tag_topk= np.zeros((max_task, max_task))

    for t, (_, ncls) in enumerate(taskcla):
        if t >= max_task:
            continue
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        net.add_head(ncls)
        net.to(device)
        appr.train(t, trn_loader[t], val_loader[t])
        print('-' * 108)
        # Test
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u], acc_tag_topk[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
                forg_tag_topk[t, u] = acc_tag_topk[:t, u].max(0) - acc_tag_topk[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} |'
                  ' TAw acc={:5.1f}%, forg={:5.1f}% |'
                  ' TAg acc={:5.1f}%, forg={:5.1f}% |'
                  ' TAg topk{} acc={:5.1f}%, forg={:5.1f}% | <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u],
                                                                 args.topk,
                                                                 100 * acc_tag_topk[t, u], 100 * acc_tag_topk[t, u]))

            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag_topk{}'.format(args.topk), group='test',
                              value=100 * acc_tag_topk[t, u])

            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag_topk{}'.format(args.topk), group='test',
                              value=100 * forg_tag_topk[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, num_tasks=args.num_tasks, name="acc_taw", step=t)
        logger.log_result(acc_tag, num_tasks=args.num_tasks, name="acc_tag", step=t)
        logger.log_result(acc_tag_topk, num_tasks=args.num_tasks,
                          name="acc_tag_topk{}".format(args.topk), step=t)

        logger.log_result(forg_taw, num_tasks=args.num_tasks, name="forg_taw", step=t)
        logger.log_result(forg_tag, num_tasks=args.num_tasks, name="forg_tag", step=t)
        logger.log_result(forg_tag_topk, num_tasks=args.num_tasks,
                          name="forg_tag_topk{}".format(args.topk), step=t)

        logger.save_model(net.state_dict(), num_tasks=args.num_tasks, task=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), num_tasks=args.num_tasks, name="avg_accs_taw", step=t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), num_tasks=args.num_tasks, name="avg_accs_tag", step=t)
        logger.log_result(acc_tag_topk.sum(1) / np.tril(np.ones(acc_tag_topk.shape[0])).sum(1),
                          num_tasks=args.num_tasks, name="avg_accs_tag_topk{}".format(args.topk), step=t)

        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), num_tasks=args.num_tasks, name="wavg_accs_taw", step=t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), num_tasks=args.num_tasks, name="wavg_accs_tag", step=t)
        logger.log_result((acc_tag_topk * aux).sum(1) / aux.sum(1),
                          num_tasks=args.num_tasks, name="wavg_accs_tag_topk{}".format(args.topk), step=t)

    # save mean feat
    if hasattr(appr, 'exemplar_means'):
        path = os.path.join(args.results_path, full_exp_name, 'models', '{}tasks'.format(args.num_tasks),'mean_feat.npy')
        means = np.array(torch.stack(appr.exemplar_means).cpu())
        np.save(path, means)


if __name__ == '__main__':
    train()

