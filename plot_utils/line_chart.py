import matplotlib.pyplot as plt
import numpy as np


def plot_line_chart(xlabels, ylabels, dir_path, color='red', label=None, title='line chart', xlabel_name='snrs',
                    ylabel_name='acc'):
    """
    :param xlabels:     (array) x轴数据
    :param ylabels:     (array) y轴数据
    :param dir_path:    (string) 保存地址
    :param color:       (string) 颜色
                        Default: 'red'
    :param label:       (list) 折线的名称
                        Default: None
    :param title:       (string)  图像标题
                        Default: 'line chart'
    :param xlabel_name: (string) x轴的标签
                        Default: 'snrs'
    :param ylabel_name: (string): y轴的标签
                        Default: 'acc'
    :return:
    """

    # Plot accuracy curve
    if label is None:
        label = []
    fig, ax = plt.subplots(1, 1)
    ax.plot(xlabels, ylabels, color=color, label=label)
    # ax.set_xlim(-2, 20)
    if xlabels.shape[0] < 20:
        # 若x轴标签数较少，逐个打印
        ax.set_xticks(xlabels)
    plt.grid(True, linestyle='--', alpha=1)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)

    plt.legend()  # 显示图例
    plt.title(title)
    plt.savefig(dir_path+title+'.png')
    plt.show()


def plot_multiple_line_chart(xlabels, ylabels, dir_path, color=None, label=None, title='mul line chart',
                             xlabel_name='snrs', ylabel_name='acc', flag=False):
    """
    绘制多条折线图, 根据输入数据添加plot函数
    :param xlabels:     (list) x轴数据  list 更方便, 适应x轴数据长度不同的情况
    :param ylabels:     (list) y轴数据
    :param dir_path:    (string) 保存地址
    :param color:       (string) 颜色
                            Default: None
    :param label:       (list) 折线的名称
                            Default: None
    :param title:       (string) 图像标题
                            Default: 'mul line chart'
    :param xlabel_name: (string) x轴的标签
                            Default: 'snrs'
    :param ylabel_name: (string) y轴的标签
                            Default: 'acc'
    :param flag:         (bool) 是否局部放大
                            Default: False

    :return:
    """

    # Plot accuracy curve

    if color is None:
        color = ['red']
    if label is None:
        label = []
    fig, ax = plt.subplots(1, 1)
    # ***********更改此处****************
    ax.plot(xlabels[0], ylabels[0], color=color[0], label=label[0])
    ax.plot(xlabels[1], ylabels[1], color=color[1], label=label[1])
    # ***********更改此处****************
    ax.set_xticks(xlabels[0])
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.grid(True, linestyle='--', alpha=1)
    plt.legend()  # 显示图例
    plt.title(title)
    if flag:
        # 绘制缩放图
        axins = ax.inset_axes((0.45, 0.6, 0.5, 0.2))
        # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
        axins.plot(xlabels[0], ylabels[0], color=color[0], label=label[0])
        axins.plot(xlabels[1], ylabels[1], color=color[1], label=label[1])
        # 局部显示并且进行连线
        zone_and_linked(ax, axins, 7, 14, xlabels[0], ylabels, linked='bottom')

    plt.savefig(dir_path+title+'.png')
    plt.show()


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black", alpha=1)
    '''
    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = patches.ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = patches.ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    '''
