import random
import numpy as np
from PIL import Image
from numpy.core.shape_base import block
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data, class_indices=None, transform_x=None, transform_y=None):
        super(MemoryDataset, self).__init__()
        self.X = data['x']
        self.Y = data['y']
        self.class_indices = class_indices
        # transform 在data_loader.py里面处理过了，这里应该没用
        self.transform_x = transform_x
        self.transform_y = transform_y
        if self.transform_x:
            self.X = self.transform_x(self.X)
        if self.transform_y:
            self.Y = self.transform_y(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.Y)


def get_data(trn_data, tst_data, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """
    Prepare data: dataset splits, task partition, class order
    :param trn_data:    {'x': trn_x, 'y': trn_y} 训练数据
    :param tst_data:    {'x': tst_x, 'y': tst_y} 测试数据
    :param num_tasks:   int 任务数
    :param nc_first_task:   int 第一个任务的类数量
    :param validation:  float 验证集比例
    :param shuffle_classes: bool 打乱顺序(True)
    :param class_order: []  标签  default: None
    :return:
    data:   {'tt': {'name':, 'trn':{'x':[], 'y':[]}, 'val':, 'tst':, 'ncla':}}
            'tt': int 任务序号
                'name': string 任务名
                'trn', 'val', 'tst': 训练集, 验证集, 测试集
                    'x': [] 样本数据
                    'y': [] 标签
                'ncla': int 每个任务的类数量
    taskcla:    [(tt, ncla)]    任务详情
    class_order:    [] 标签
    """

    data = {}
    taskcla = []
    clsanalysis = {}
    testclsanalysis = {}
    list_of_clsnum = []
    inorder = True
    if class_order is None:
        num_classes = len(np.unique(trn_data['y']))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        # 依次添加未除尽的样本
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        clsanalysis[tt] = np.zeros(cpertask[tt])
        testclsanalysis[tt] = np.zeros(cpertask[tt])

    # training set analysis

    # labelcounter=np.zeros(num_classes)
    # ALL OR TRAIN
    filtering = np.isin(trn_data['y'], class_order)
    if filtering.sum() != len(trn_data['y']):
        trn_data['x'] = trn_data['x'][filtering]
        trn_data['y'] = np.array(trn_data['y'])[filtering]

    # print(order)
    for this_image, this_label in zip(trn_data['x'], trn_data['y']):
        # If shuffling is false, it won't change the class number
        # 若class_order 被打乱, 需要重新获取标签
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        # 根据标签将数据加入相应的任务
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        # 每个任务中的标签都是从 0 开始
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])
        # 计算每一类的数量
        clsanalysis[this_task][this_label - init_class[this_task]] += 1

    # ALL OR TEST
    filtering = np.isin(tst_data['y'], class_order)
    if filtering.sum() != len(tst_data['y']):
        tst_data['x'] = tst_data['x'][filtering]
        tst_data['y'] = tst_data['y'][filtering]
    for this_image, this_label in zip(tst_data['x'], tst_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # this_label = order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])
        testclsanalysis[this_task][this_label - init_class[this_task]] += 1
    # clsanalyze(clsanalysis,testclsanalysis)
    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        # 每个任务类的数量验证
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                if int(np.round(len(cls_idx) * validation)) == 0:
                    rnd_img = [0]    # 保证起码有一个样本
                else:
                    rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # convert them to numpy arrays
    for tt in data.keys():
        for split in ['trn', 'val', 'tst']:
            data[tt][split]['x'] = np.asarray(data[tt][split]['x'])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order




