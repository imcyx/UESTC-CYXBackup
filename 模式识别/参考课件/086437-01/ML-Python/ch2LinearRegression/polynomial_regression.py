# -*- coding: utf-8 -*-
"""
多项式回归拟合历届奥运会自由泳100米数据

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""
from typing import List

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
        content: List[str] = fr.readlines()
        x = [f.split(",")[: -1] for f in content]
        y = [float(f.split(",")[-1].strip("\n")) for f in content]
    return x, y


def normal_equation(x, y):
    """
    正规方程求解theta参数集
    输入
        x：特征矩阵，y：目标属性
    输出
        theta：参数集，向量
    """
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    return theta


def plot_scatter(x, y, theta):
    """
    绘制散点图
    输入参数：
        x：一个属性，y：目标属性，theta：参数
    返回：
        无
    """
    import matplotlib as mpl

    # 防止plt汉字乱码
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure()
    plt.scatter(x, y, c='b', marker='o', s=5, linewidths=2)
    lim = np.array([min(x) - 4, max(x) + 4])  # 为上下限留一点空隙
    plt.xlim(lim)  # 设定x坐标轴的范围
    # 绘制拟合线
    rank: int = len(theta) - 1
    # 显示的数据点。左右留边距2
    p_x = np.arange(min(x) - 2, max(x) + 2, 0.01).reshape(-1, 1)
    plot_x = np.ones((len(p_x), 1))
    for i in range(rank):
        plot_x = np.column_stack((plot_x, np.power(p_x, i+1)))
    plt.plot(p_x, np.dot(plot_x, theta).reshape(-1, 1), 'r')
    plt.xlabel(u'奥运会举办年编号')
    plt.ylabel(u'取胜时间（秒）')
    plt.title(u'%d次模型' % rank)

    plt.show()


def poly_regression(x, y, rank):
    """
    绘制散点图
    输入参数：
        x：一个属性，y：目标属性，rank：多项式阶次
    返回：
        无
    """
    n = len(y)  # 样本数
    x_data = np.ones((n, 1))
    for i in range(rank):
        x_data = np.column_stack((x_data, np.power(x, i+1)))

    theta = normal_equation(x_data, y)
    # 绘图
    plot_scatter(x, y, theta)


def main():
    #    加载奥运会数据
    file_path = "../data/Freestyle100m.csv"
    x, y = read_csv(file_path)

    x = [float(f[0]) for f in x]
    # 转换为Numpy数组
    x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
    # 将奥运会举办年转换为届，避免数值计算问题
    x -= x[0]
    x /= 4

    poly_regression(x, y, 1)
    poly_regression(x, y, 2)
    poly_regression(x, y, 4)
    poly_regression(x, y, 8)


if __name__ == "__main__":
    main()
