# -*- coding: utf-8 -*-
"""
绘制历届奥运会自由泳100米记录的数据

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt


def read_csv(file):
    """
    读取CSV文件，允许有多个属性，文件最后一列为目标属性
    输入参数：
        file：文件名
    返回：
        x：属性，y：目标属性
    """
    with open(file, encoding="utf-8") as fr:
        content = fr.readlines()
        x = [f.split(",")[: -1] for f in content]
        y = [float(f.split(",")[-1].strip("\n")) for f in content]
    return x, y


def plot_scatter(x, y):
    """
    绘制散点图
    输入参数：
        x：只绘制最开始一个属性，y：目标属性
    返回：
        无
    """
    import matplotlib as mpl

    # 防止plt汉字乱码
    mpl.rcParams[u'font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    x1 = [float(f[0]) for f in x]

    # plt.figure()

    plt.scatter(x1, y, c='b', marker='o', s=5, linewidths=2)
    mylim = np.array([min(x1) - 4, max(x1) + 4])     # 为上下限留一点空隙
    plt.xlim(mylim)          # 设定x坐标轴的范围
    plt.xlabel(u'奥运会年')
    plt.ylabel(u'取胜时间（秒）')
    plt.show()


# %%
if __name__ == "__main__":
    file_path = "../data/Freestyle100m.csv"
    X, Y = read_csv(file_path)
    plot_scatter(X, Y)
