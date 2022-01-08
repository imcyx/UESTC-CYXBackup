# -*- coding: utf-8 -*-
"""
使用高斯判别模型求解分类问题

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def plot_decision_boundary(x, y, mu, sigma):
    """
    绘制二元分类问题的决策边界
    输入参数
        x：输入，y：输出，mu：均值，sigma：协方差
    输出参数
        无
    """
    plt.figure()
    neg_x = x[np.where(y[:, 0] == 0)]
    pos_x = x[np.where(y[:, 0] == 1)]
    neg = plt.scatter(neg_x[:, 0], neg_x[:, 1], c='b', marker='o')
    pos = plt.scatter(pos_x[:, 0], pos_x[:, 1], c='r', marker='+')

    # 绘制决策边界
    # 网格范围
    u = np.linspace(min(x[:, 0]), max(x[:, 0]), 150)
    v = np.linspace(min(x[:, 1]), max(x[:, 1]), 150)
    uu, vv = np.meshgrid(u, v)  # 生成网格数据
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    sqrt_det_sigma = np.sqrt(np.linalg.det(sigma))
    inv_sigma = np.linalg.pinv(sigma)
    head = 1 / (2 * np.pi * sqrt_det_sigma)
    z0 = np.zeros(len(uv))
    z1 = np.zeros(len(uv))
    for i in range(len(uv)):
        z0[i] = head * np.exp(-1 / 2 * ((uv[i] - mu[0]) @ inv_sigma @ (uv[i] - mu[0]).T))
        z1[i] = head * np.exp(-1 / 2 * ((uv[i] - mu[1]) @ inv_sigma @ (uv[i] - mu[1]).T))
    # 保持维度一致
    z0 = z0.reshape(uu.shape)
    z1 = z1.reshape(uu.shape)
    z_db = z1 - z0
    # 画图
    plt.contour(uu, vv, z0)
    plt.contour(uu, vv, z1)
    plt.contour(uu, vv, z_db, 0)

    # 坐标
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 图例
    plt.legend([neg, pos], ['负例', '正例'], loc='lower right')
    plt.show()


def main():
    np.random.seed(0)
    # 二个高斯
    k = 2
    real_phi = 0.6
    real_means = [[0.0, -3.2], [1.0, 3.5]]
    real_covs = np.zeros((2, 2, 2))
    real_covs[0] = [[0.58, -0.05], [-0.05, 1.55]]
    real_covs[1] = [[0.65, -0.15], [-0.15, 1.12]]
    priors = [1 - real_phi, real_phi]

    n = 100
    x = np.zeros((n, 2))
    y = np.zeros((n, 1))

    # 选择一个高斯，并随机抽样数据
    for i in range(n):
        comp = np.random.choice(2, p=priors)
        x[i] = np.random.multivariate_normal(real_means[comp], real_covs[comp], 1)
        y[i] = comp

    # 估计参数
    neg_idx = np.squeeze(np.where(y[:, 0] == 0))
    pos_idx = np.squeeze(np.where(y[:, 0] == 1))
    phi = len(pos_idx) / n

    mu = np.zeros((2, 2))
    mu[0] = np.sum(x[neg_idx], axis=0) / len(neg_idx)
    mu[1] = np.sum(x[pos_idx], axis=0) / len(pos_idx)
    sigma = np.zeros((k, k))
    for i in range(n):
        x_i = (x[i] - mu[int(y[i, 0])]).reshape(1, -1)
        sigma += np.dot(x_i.T, x_i)
    sigma /= n

    print('估计出来的参数：\n')
    print(f'phi：{phi}\n')
    print(f'mu0：{mu[0]}\n')
    print(f'mu1：{mu[1]}\n')
    print(f'sigma：{sigma}\n')

    # 绘制决策边界
    plot_decision_boundary(x, y, mu, sigma)


if __name__ == "__main__":
    main()
