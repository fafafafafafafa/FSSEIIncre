import argparse
import datasets.memory_dataset as memd
from datasets.dataset_config import dataset_config
from torch.utils import data
import numpy as np


def normalize_data(X):
    X = np.array(X)
    print('X: ', X.shape)
    min_value = X.min()
    max_value = X.max()
    X = (X - min_value) / (max_value - min_value)
    X = np.float32(X)
    return X


def get_datasets(dataset, train_path, test_path, num_tasks, nc_first_task, validation, class_order=None):
    """
    Extract datasets and create Dataset class
    :param dataset: [] 数据集
    :param train_path:  {'x':, 'y':}
    :param test_path:   {'x':, 'y':}
    :param num_tasks:   int 任务数
    :param nc_first_task:   int 第一个任务的类数量
    :param validation:  float 验证集比例
    :param class_order: []  标签  default: None
    :return:
        trn_dset:   [MemoryDataset, ...]
        val_dset:   [MemoryDataset, ...]
        tst_dset:   [MemoryDataset, ...]
        taskcla:    [(tt, ncla)]    任务详情
    """

    trn_dset, val_dset, tst_dset = [], [], []
    if 'fs_sei' in dataset:
        trn_x = np.load(train_path['x'])
        trn_y = np.load(train_path['y'])
        tst_x = np.load(test_path['x'])
        tst_y = np.load(test_path['y'])

        trn_x = normalize_data(trn_x)
        tst_x = normalize_data(tst_x)
        trn_y = trn_y.astype(np.uint8)
        tst_y = tst_y.astype(np.uint8)

        trn_data = {'x': trn_x, 'y': trn_y}
        tst_data = {'x': tst_x, 'y': tst_y}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset
    else:
        raise ValueError("dataset is not found")
    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], class_indices))
        val_dset.append(Dataset(all_data[task]['val'], class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], class_indices))
        offset += taskcla[task][1]
    print(len(trn_dset))
    print(len(val_dset))
    print(len(tst_dset))
    return trn_dset, val_dset, tst_dset, taskcla


def get_loaders(datasets, num_tasks, nc_first_task, batch_size, num_workers, pin_memory, validation=.1):
    """
        Apply transformations to Datasets and create the DataLoaders for each task
    :param datasets:    ['dataset_name1', ...] 数据集名称
    :param num_tasks:   int 任务数
    :param nc_first_task:   int 第一个任务的类数量
    :param batch_size:  int
    :param num_workers: int 有多少个进程用于数据加载
    :param pin_memory:  bool    将数据转移到GPU上(True)
    :param validation:  float 验证集比例
    :return:
        trn_loader:   [DataLoader, ...]
        val_loader:   [DataLoader, ...]
        tst_loader:   [DataLoader, ...]
        taskcla:    [(tt, ncla)]    任务详情
    """
    trn_loader, val_loader, tst_loader = [], [], []
    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # transformations
        '''      
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      normalize=dc['normalize'],
                                                      extend_channel=dc['extend_channel'])
        '''


        # datasets
        trn_dset, val_dset, tst_dset, curtaskcla = get_datasets(cur_dataset,
                                                                dc['train_path'],
                                                                dc['test_path'],
                                                                num_tasks,
                                                                nc_first_task,
                                                                validation=validation
                                                                )
        # apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].Y = [elem + dataset_offset for elem in trn_dset[tt].Y]
                val_dset[tt].Y = [elem + dataset_offset for elem in val_dset[tt].Y]
                tst_dset[tt].Y = [elem + dataset_offset for elem in tst_dset[tt].Y]
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_loader.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=pin_memory))
            val_loader.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
            tst_loader.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
    return trn_loader, val_loader, tst_loader, taskcla


if __name__ == '__main__':
    pass
