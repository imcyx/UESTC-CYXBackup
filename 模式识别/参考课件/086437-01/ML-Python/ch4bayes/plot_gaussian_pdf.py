# -*- coding: utf-8 -*-
"""
本示例展示如何调用np.random.multivariate_normal函数随机抽样高斯分布
总共有8个高斯分布，均值和方差不同

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def plot_data(mu, sigma):
    """ 绘制二维数据集散点图 """
    n = 600         # 样本数
    lim = [-7, 7]  # 显示坐标范围
    x = np.random.multivariate_normal(mu, sigma, n)
    plt.figure(figsize=(4, 4))
    plt.scatter(x[:, 0], x[:, 1], c='k', marker='.')
    plt.axis('equal')
    plt.xlim(lim)
    plt.ylim(lim)
    plt.show()


def main():
    np.random.seed(0)

    # 1
    mu = [0, 0]
    sigma = [[1, 0], [0, 1]]
    plot_data(mu, sigma)

    # 2
    mu = [0, 0]
    sigma = [[0.2, 0], [0, 0.2]]
    plot_data(mu, sigma)

    # 3
    mu = [0, 0]
    sigma = [[2, 0], [0, 2]]
    plot_data(mu, sigma)

    # 4
    mu = [0, 0]
    sigma = [[0.2, 0], [0, 2]]
    plot_data(mu, sigma)

    # 5
    mu = [0, 0]
    sigma = [[2, 0], [0, 0.2]]
    plot_data(mu, sigma)

    # 6
    mu = [0, 0]
    sigma = [[1, 0.5], [0.5, 1]]
    plot_data(mu, sigma)

    # 7
    mu = [0, 0]
    sigma = [[0.3, 0.5], [0.5, 2]]
    plot_data(mu, sigma)

    # 8
    mu = [0, 0]
    sigma = [[0.3, -0.5], [-0.5, 2]]
    plot_data(mu, sigma)


if __name__ == "__main__":
    main()

