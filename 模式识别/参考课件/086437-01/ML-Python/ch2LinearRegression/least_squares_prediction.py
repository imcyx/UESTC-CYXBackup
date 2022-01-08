# -*- coding: utf-8 -*-
"""
使用最小二乘法，预测未来两届的奥运会自由泳100米记录

% The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
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


def plot_scatter(x, y, w0, w1, pred_x, pred_y):
    """
    绘制散点图
    输入参数：
        x：一个属性，y：目标属性，w0：截距，w1：斜率，pred_x：预测x，pred_y：预测y
    返回：
        无
    """
    import matplotlib as mpl

    # 防止plt汉字乱码
    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure()
    plt.scatter(x, y, c='b', marker='o', s=5, linewidths=2)
    plt.scatter(pred_x, pred_y, c='g', marker='^')
    lim = np.array([min(x) - 4, max(pred_x) + 4])  # 为上下限留一点空隙
    plt.xlim(lim)  # 设定x坐标轴的范围
    plt.plot(lim, w0 + w1 * lim, 'r')  # 绘制拟合直线
    plt.xlabel(u'奥运会年')
    plt.ylabel(u'取胜时间（秒）')

    plt.show()


def main():
    #    加载奥运会数据
    file_path = "../data/Freestyle100m.csv"
    x, y = read_csv(file_path)

    x = [float(f[0]) for f in x]
    x, y = np.array(x), np.array(y)  # 转换为Numpy数组
    n = len(y)  # 样本数

    # 计算均值
    mu_x = np.sum(x) / n
    mu_y = np.sum(y) / n
    mu_xy = np.sum(np.multiply(y, x)) / n
    mu_xx = np.sum(np.multiply(x, x)) / n

    # 计算 w1(斜率)
    w1 = (mu_xy - mu_x * mu_y) / (mu_xx - mu_x ** 2)
    # 计算 w0(截距)
    w0 = mu_y - w1 * mu_x

    print(f"w0 = {w0}, w1 = {w1}")
    # 预测2020和2024年的记录
    new_x = np.array([2020, 2024])  # 加入预测年
    new_y = w0 + w1 * new_x  # 预测
    print(f"预测y = {new_y}")

    # 绘图
    plot_scatter(x, y, w0, w1, new_x, new_y)


if __name__ == "__main__":
    main()
