# -*- coding: utf-8 -*-
"""
从混合高斯中生成数据

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


def sampling(prob_distribution):
    """
     输入
        probDistribution ： 概率分布
    输出
        idx ： 抽样到的类别索引
    """
    idx = np.random.choice(len(prob_distribution), p=prob_distribution)
    return idx


def plot_mixture_gaussians(x, mu, sigma):
    """
    绘制混合高斯
    输入参数
        x：输入，mu：均值，sigma：协方差
    输出参数
        无
    """
    plt.clf()
    left, right = -3, 6
    bottom, top = -6, 6
    plt.scatter(x[:, 0], x[:, 1], c='b', marker='o')

    # 绘制高斯轮廓图
    # 网格范围
    u = np.linspace(left, right, 150)
    v = np.linspace(bottom, top, 150)
    uu, vv = np.meshgrid(u, v)  # 生成网格数据
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    sqrt_det_sigma0 = np.sqrt(np.linalg.det(sigma[0]))
    inv_sigma0 = np.linalg.pinv(sigma[0])
    head0 = 1 / (2 * np.pi * sqrt_det_sigma0)
    sqrt_det_sigma1 = np.sqrt(np.linalg.det(sigma[1]))
    inv_sigma1 = np.linalg.pinv(sigma[1])
    head1 = 1 / (2 * np.pi * sqrt_det_sigma1)
    z0 = np.zeros(len(uv))
    z1 = np.zeros(len(uv))
    for i in range(len(uv)):
        z0[i] = head0 * np.exp(-1 / 2 * ((uv[i] - mu[0]) @ inv_sigma0 @ (uv[i] - mu[0]).T))
        z1[i] = head1 * np.exp(-1 / 2 * ((uv[i] - mu[1]) @ inv_sigma1 @ (uv[i] - mu[1]).T))
    # 保持维度一致
    z0 = z0.reshape(uu.shape)
    z1 = z1.reshape(uu.shape)
    # 画图
    plt.contour(uu, vv, z0)
    plt.contour(uu, vv, z1)

    # 坐标
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((left, right))
    plt.ylim((bottom, top))
    plt.pause(3)


def main():
    # 定义混合成分
    means = np.array([[3, 3], [1, -3]])
    covs = np.zeros((2, 2, 2))
    covs[0] = np.array([[1, 0], [0, 1.5]])
    covs[1] = np.array([[2, -1], [-1, 1]])
    pis = np.array([0.4, 0.6])

    # 一次生成一个数据点
    np.random.seed(1234)
    n = 50      # 数据点总数
    plot_points = np.arange(9, n, 10)     # 每隔10个点暂停一次
    comp = sampling(pis)
    x = np.random.multivariate_normal(means[comp], covs[comp], 1)
    plt.figure()
    # 打开交互模式
    plt.ion()
    for i in range(1, n):
        # 从先验中选择并投掷一枚有偏的硬币
        # 选择一个高斯
        comp = sampling(pis)
        x = np.append(x, np.random.multivariate_normal(means[comp], covs[comp], 1), axis=0)
        if i in plot_points:
            # 绘制
            plot_mixture_gaussians(x, means, covs)

    # 关闭交互模式
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
