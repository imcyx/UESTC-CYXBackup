# -*- coding: utf-8 -*-
"""
PCA示例3
对MNIST数据集进行PCA分析

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.append('..')
from utils.mnist_read import load_mnist

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def display_mnist_data(x):
    """
    显示MNIST图像数据，默认图像高宽相等
    输入:
        x：要显示的图像样本
    输出:
        无
    """
    figsize = (8, 8)
    # 计算输入数据的行数和列数
    if x.ndim == 2:
        n, d = x.shape
    elif x.ndim == 1:
        d = x.size
        n = 1
        x = x.reshape(1, -1)
    else:
        raise IndexError('输入只能是一维或二维的图片样本集合。')

    img_width = int(np.round(np.sqrt(d)))
    img_height = int(d / img_width)

    # 计算要显示的图像的行数和列数
    display_rows = int(np.floor(np.sqrt(n)))
    display_cols = int(np.ceil(n / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    ax_array = [ax_array] if n == 1 else ax_array.ravel()

    # 循环显示图像
    for i, ax in enumerate(ax_array):
        ax.imshow(x[i].reshape(img_height, img_width, order='C'), cmap='gray')
        ax.axis('off')

    plt.show()


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
    # norm_x = (x - mu) / (sigma + 1e-6)
    norm_x = (x - mu)
    return norm_x, mu, sigma


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
    x, y = load_mnist('../data/mnist/')
    display_mnist_data(x[:100])

    norm_x, mu, sigma = feature_normalize(x)
    u, s = pca(norm_x)

    # 可视化前36个主成分
    display_mnist_data(u[:, :100].T)

    # 投影到k维空间
    k = 100
    z = project_data(norm_x, u, k)
    print('降维后的数据大小：{}\n'.format(z.shape))

    # 计算只使用前K个成分，投影均方误差与方差的比值
    ratio = np.sum(s[:k]) / np.sum(s)
    print('只使用前{}个成分，投影均方误差与方差的比值:{:.2%}\n\n'.format(k, ratio))

    # 重建数据
    rec_x = recover_data(z, u, k)
    # 绘制原始数据和重建数据
    display_mnist_data(norm_x[:100])
    display_mnist_data(rec_x[:100])


if __name__ == "__main__":
    main()
