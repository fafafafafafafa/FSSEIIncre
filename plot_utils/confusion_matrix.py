import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, dir_path, title='Confusion matrix', cmap=plt.cm.Blues, labels=None, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    混淆矩阵
    参数说明:
        输入:
            cm (narray): 图像数据  行为正确模式, 列为预测模式
            dir_path (string): 保存地址
            title (string):  图像标题
                            Default: 'Confusion matrix'
            cmap : 颜色图实例或注册的颜色图名称
                    Default: plt.cm.Blues
            labels (list): 坐标
                    Default: None
            normalize (bool): 是否归一化
                    Default: False
        输出:

    """
    if labels is None:
        labels = []
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # cm
    plt.figure(figsize=(6, 6))  # 设置图的尺寸大小
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()

    tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels, rotation=45, fontsize=12)

    if cm.shape[0] < 10:
        plt.xticks(tick_marks, labels, fontsize=12)
        plt.yticks(tick_marks, labels, fontsize=12)
        fmt = '.2f' if normalize else '.3f'
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2  # 设定矩阵颜色阈值
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # i,j 代表 cm的行和列数，plt.text 要求输入 坐标(j, i)
            plt.text(j, i, format(cm[i, j], fmt),
                     verticalalignment="center",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    '''
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    '''
    plt.savefig(dir_path + '/%s.png' % title, dpi=1000, format='png')
    print('cm save in ', dir_path + '/%s' % title)
    plt.close()