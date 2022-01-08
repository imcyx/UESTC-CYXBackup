# -*- coding: utf-8 -*-
"""
练习参考答案
本示例与pca_demo2.py的区别是数据有四个簇，有两个主成分的本征值较大

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
    随机生成七维数据，其中，二维方差较大，五维是方差较小
    输入:
        无
    输出:
        x_data：随机生成的数据，y_data：仅用于绘制的标签
    """
    np.random.seed(12)
    n = 20

    # 随机产生数据集
    x = np.row_stack((np.random.randn(n, 2), np.random.randn(n, 2) + 5,
                      np.random.randn(n, 2) + np.array([5, 0]), np.random.randn(n, 2) + np.array([0, 5])))
    # 添加5个随机维
    x_data = np.column_stack((x, np.random.randn(len(x), 5)))

    # 仅用于绘制的标签
    y_data = np.row_stack((np.zeros((n, 1)), np.ones((n, 1)), 2 * np.ones((n, 1)), 3 * np.ones((n, 1))))
    return x_data, y_data


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
    # norm_x = (x - mu) / sigma
    norm_x = (x - mu)
    return norm_x, mu, sigma


def plot_data(x, y, title):
    """
    绘制数据散点图
    输入:
        x：数据，y：绘图用的标签，title：
    输出:
        无
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    marks = ['ro', 'gs', 'bd', 'k^']
    for k in range(4):
        plot_x = x[np.where(y[:, 0] == k)]
        ax.plot(plot_x[:, 0], plot_x[:, 1], marks[k])
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()


def plot_eig_vectors(x, y, u):
    """
    绘制散点图，绘制本征向量箭头
    输入:
        x：数据，y：绘图用的标签，u：本征向量组成的矩阵
    输出:
        无
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    marks = ['ro', 'gs', 'bd', 'k^']
    for k in range(4):
        plot_x = x[np.where(y[:, 0] == k)]
        ax.plot(plot_x[:, 0], plot_x[:, 1], marks[k])
    ax.axis([-6, 6, -6, 6])
    ax.set_aspect('equal')

    xl = np.array([-6, 6])
    for i in range(2):
        ax.plot(xl, xl * u[0, i] / u[1, i], 'k')
    plt.title(u'两个成分方向')
    plt.show()


def plot_eig_values_bar(s):
    """
    绘制本征值的条形图
    输入:
        s：本征值组成的向量
    输出:
        无
    """
    plt.figure()
    plt.bar(range(len(s)), s)
    plt.xlabel(u'投影维数')
    plt.ylabel(u'方差')
    plt.title(u'本征值条形图')
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
    x, y = generate_data()
    plot_data(x, y, u'原始数据')

    norm_x, mu, sigma = feature_normalize(x)
    u, s = pca(norm_x)
    # 绘制本征向量方向
    plot_eig_vectors(norm_x, y, u)

    # 绘制本征值条形图
    plot_eig_values_bar(s)

    # 计算只使用两个成分，投影均方误差与方差的比值
    ratio = (s[0] + s[1]) / sum(s)
    print('只使用两个成分，投影均方误差与方差的比值:{:.2%}\n'.format(ratio))

    # 投影到二维空间
    k = 2
    z = project_data(norm_x, u, k)
    plot_data(z, y, u'投影到前两个成分的数据')

    # 重建数据
    rec_x = recover_data(z, u, k)
    # 绘制重建数据
    plot_data(rec_x, y, u'重建原始高维数据的近似')


if __name__ == "__main__":
    main()
