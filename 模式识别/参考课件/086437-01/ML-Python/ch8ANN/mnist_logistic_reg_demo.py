# -*- coding: utf-8 -*-
"""
逻辑回归（Logistic Regression）分类
这里测试多元分类，使用一对多方式

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.append('..')
from utils.mnist_read import load_mnist

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def sigmoid(z):
    """ S型激活函数 """
    g = 1 / (1 + np.exp(-z))
    return g


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


def compute_cost(theta, x, y, my_lambda):
    """
    计算逻辑回归的代价
    使用theta作为逻辑回归的参数，计算代价J
    输入参数
        theta：参数，x：输入，y：输出，my_lambda：正则化参数
    输出参数
        j_value：计算的J值， grad：梯度
    """
    n = len(y)  # 训练样本数

    temp_theta = theta.copy()
    temp_theta[0] = 0  # 正则化不包括截距项

    h = sigmoid(np.dot(x, theta))
    eps = 1e-6      # 避免数值计算问题
    j_value = -1 / n * np.add(np.dot(y.T, np.log(h + eps)), np.dot((1 - y).T, np.log(1 - h + eps))) + \
              my_lambda / 2 * np.dot(temp_theta.T, temp_theta)
    grad = 1 / n * np.dot(x.T, np.subtract(h, y)) + my_lambda * temp_theta
    return j_value, grad


def gradient_descent(x, y, init_theta, my_lambda):
    """
    梯度下降函数，找到合适的参数
    输入参数
        x：输入，y：输出，init_theta：参数，my_lambda：正则化参数
    输出参数
        theta：学习到的参数
    """
    # 优化
    options = {'maxiter': 200}
    res = optimize.minimize(lambda p: compute_cost(p, x, y, my_lambda), init_theta, jac=True, method='TNC',
                            options=options)
    theta = res.x

    return theta


def predict(theta, new_x):
    """
    使用学习到的逻辑回归参数theta来预测新样本new_x的标签
    输入参数：
        theta：逻辑回归参数，new_x：新样本集
    返回：
        y_hat：预测的十种类别之一
    """
    pred = sigmoid(np.dot(new_x, theta))
    y_hat = np.argmax(pred, axis=1)
    return y_hat


def main():
    # 正则化参数lambda
    my_lambda = 0.001

    #  加载MNIST数据
    x, y = load_mnist('../data/mnist/')
    display_mnist_data(x[:100])

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    # 规范化
    x_data /= 255

    n = len(y)  # 样本数

    # 添加一列全1，以扩展x
    x_data = np.column_stack((np.ones((n, 1)), x_data))

    d = len(x_data[0])

    # 10个二元分类器的theta
    theta = np.zeros((d, 10))

    # 转换为二元分类问题，分别计算代价和梯度
    for classifier_idx in range(10):
        temp_y = np.zeros(n)
        temp_y[np.where(y == classifier_idx)] = 1

        # 梯度下降
        init_theta = np.zeros(d)
        temp_theta = gradient_descent(x_data, temp_y, init_theta, my_lambda)

        theta[:, classifier_idx] = temp_theta

    # 计算在训练集上的分类正确率
    y_hat = predict(theta, x_data)
    print('训练集上的分类正确率: {:.2%}'.format(np.mean(y_hat == y)))


if __name__ == "__main__":
    main()
