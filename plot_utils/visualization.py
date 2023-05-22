import matplotlib
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np


def plot_visual(data, targets, dir_path, labels):
    """
     :param data: (array) 输入数据 (batch_size, embedding_size)
     :param targets: (array)  标签 (batch_size,)
     :param dir_path: (string)  图片存储地址

     :return:
     """

    # 设置matplotlib正常显示中文和负号
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 设置中文宋体、西文NEW Roma
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    plt.rcParams.update({'font.size': 18})

    fig = plt.figure(figsize=(6, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    ax = plt.subplot(111)  # 创建子图，经过验证111正合适，尽量不要修改
    # 控制边框
    plt.axis('on')

    print(data.shape)
    print(targets.shape)
    # targets = torch.unsqueeze(targets, 1)
    data = np.array(data)
    targets = np.array(targets)
    print(data.shape)
    print(targets.shape)
    tsne = TSNE(n_components=2, init='pca')
    data_tsne = tsne.fit_transform(data)
    x_min, x_max = np.min(data_tsne, 0), np.max(data_tsne, 0)
    data_tsne = (data_tsne - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    # plt.cm.Set31(idx) 0-11 12中颜色
    markers = ['.', 'X', 'v', '^', '<', '>', '8', 's', 'p', '*'] # 10
    # 一共12*10种类型
    num_marker = -1
    for idx, label in enumerate(labels):
        idx_label = np.where(targets == label)
        if idx % 12 == 0:
            num_marker += 1
        plt.scatter(data_tsne[idx_label, 0], data_tsne[idx_label, 1],
                    s=100, marker=markers[num_marker], color=plt.cm.Set3(idx % 12), label=label)

    # plt.colorbar()
    if len(labels) <= 10:
        plt.legend(loc='upper right', handlelength=1, frameon=False, labelspacing=0.4)  # 原来的size=3
    '''
    x_min, x_max = np.min(data_tsne, 0), np.max(data_tsne, 0)
    data_tsne = (data_tsne-x_min) / (x_max-x_min)
    for i in range(data_tsne.shape[0]):
        plt.text(data_tsne[i, 0], data_tsne[i, 1], str(targets[i]),
                 color=plt.cm.Set1(targets[i]/90.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    '''

    plt.savefig(dir_path, dpi=1000, format='png')
    return fig
    # plt.show()


if __name__ == '__main__':
    pass