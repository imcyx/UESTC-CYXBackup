# -*- coding: utf-8 -*-
"""
PCA示例1
绘制特征向量，将二维数据投影到一维空间

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def generate_data():
    """
    随机生成二维数据
    输入:
        无
    输出:
        x：随机生成的二维数据
    """
    np.random.seed(1234)

    # 随机产生数据集
    mu = [3, 5]
    sigma = [[1, 0.5], [0.5, 0.5]]
    n = 50
    x = np.random.multivariate_normal(mu, sigma, n)
    return x


def feature_normalize(x):
    """
    特征规范化
    将特征规范化为0均值1方差
    输入:
        x：要规范化的特征
    输出:
        norm_x：已规范化的特征，mu：均值，sigma：方差
    """
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    norm_x = (x - mu) / sigma
    return norm_x, mu, sigma


def plot_eig_vectors(x, mu, u, s):
    """
    绘制二维数据散点图，绘制本征向量箭头
    输入:
        x：二维数据，mu：均值，u：本征向量组成的矩阵，s：本征值组成的向量
    输出:
        无
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(x[:, 0], x[:, 1], 'wo', ms=10, mec='b', mew=1)
    scale = 1.5
    for i in range(2):
        ax.arrow(mu[0, 0], mu[0, 1], scale * s[i] * u[0, i], scale * s[i] * u[1, i],
                 head_width=0.15, head_length=0.2, fc='r', ec='k', lw=2, zorder=1000)
    ax.axis([-0.5, 5.5, 2, 8])
    ax.set_aspect('equal')
    plt.show()


def plot_recover_data(norm_x, rec_x):
    """
    绘制重建数据
    输入:
        norm_x：规范化后的二维数据，rec_x：重建数据
    输出:
        无
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(norm_x[:, 0], norm_x[:, 1], 'bo', ms=8, mec='k', mew=1)
    ax.set_aspect('equal')
    plt.axis([-3, 3, -3, 3])

    # 绘制投影线
    ax.plot(rec_x[:, 0], rec_x[:, 1], 'ro', mec='r', mew=2, mfc='none')
    for x_norm, x_rec in zip(norm_x, rec_x):
        ax.plot([x_norm[0], x_rec[0]], [x_norm[1], x_rec[1]], '--g', lw=1)

    plt.show()


def pca(x):
    """
    主成分分析
    输入参数
        x：二维数据
    输出参数
        u：本征向量，s：本征值
    """
    n = len(x)

    sigma = np.dot(x.T, x) / n
    u, s, _ = np.linalg.svd(sigma)
    return u, s


def project_data(x, u, k):
    """
    投影数据
    输入:
        x：二维数据，u：本征向量组成的矩阵，k：投影子空间的维度
    输出:
        z：投影到前k个本征向量结果
    """
    z = np.dot(x, u[:, :k])
    return z


def recover_data(z, u, k):
    """
    重建数据
    输入:
        z：降维后的数据，u：本征向量组成的矩阵，k：保持的主成分的数量
    输出:
        rec_x：重建的数据
     """
    rec_x = np.dot(z, u[:, :k].T)
    return rec_x


def main():
    x = generate_data()
    norm_x, mu, sigma = feature_normalize(x)
    u, s = pca(norm_x)
    print('主本征向量：u[:, 0] = [{:.6f} {:.6f}]'.format(u[0, 0], u[1, 0]))
    # 绘制本征向量
    plot_eig_vectors(x, mu, u, s)

    # 投影到一维空间
    k = 1
    z = project_data(norm_x, u, k)

    # 重建数据
    rec_x = recover_data(z, u, k)
    # 绘制重建数据
    plot_recover_data(norm_x, rec_x)


if __name__ == "__main__":
    main()
