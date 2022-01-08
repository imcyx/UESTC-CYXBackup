# -*- coding: utf-8 -*-
"""
正则化与偏差方差

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
    生成数据
    """
    n = 40  # 样本数
    # 生成0到1之间的x
    x = np.linspace(0, 1, n).reshape((-1, 1))
    noise = 0.08          # 噪声强度
    # 目标
    y = np.sin(2 * np.pi * x) + noise * np.random.randn(len(x), 1)
    # 独立验证集
    ind_x = np.linspace(0, 1, 1000).reshape((-1, 1))
    ind_y = np.sin(2 * np.pi * ind_x) + noise * np.random.randn(len(ind_x), 1)
    return x, y, ind_x, ind_y


def poly_features(x, degree):
    """
    将两个特征转换为多项式特征
    例如，将x转换为1, x, x.^2, x.^3, ...
    输入
        x：n行1列特征，degree：阶次
    输出
        x_data：转换后的多项式特征
    """
    x_data = np.ones((len(x), 1))  # 截距项
    for i in range(1, degree + 1):
        x_data = np.column_stack((x_data, np.power(x, i)))

    return x_data


def normal_equation(x, y, my_lambda):
    """
    正则化的正规方程求解theta参数集
    输入
        x：特征矩阵，y：目标属性，my_lambda：正则化参数
    输出
        theta：参数集，向量
    """
    d = len(x[0]) - 1
    a = np.eye(d + 1, d + 1)
    a[0][0] = 0
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x) + np.dot(my_lambda, a)), x.T), y)
    return theta


def plot_loss(train_loss, ind_loss, lambdas):
    """
    绘制训练损失和验证损失曲线
    """
    plt.figure()
    plt.plot(lambdas, train_loss, 'b--', label=u'训练损失')
    plt.plot(lambdas, ind_loss, 'g-', label=u'验证损失')
    ax = plt.gca()
    ax.xaxis.get_major_formatter().set_powerlimits((0, 3))
    plt.xlabel(r'$\lambda$')
    plt.ylabel(u'损失')
    plt.title(r'最佳$\lambda$')
    plt.legend()
    plt.show()


def bv_train(x, y, ind_x, ind_y):
    """
    偏差方差训练
    """

    # 正则化参数lambda序列
    lambdas = np.linspace(0, 0.00002, 100)
    max_order = 9  # 最高模型阶次

    poly_x = poly_features(x, max_order)
    poly_ind_x = poly_features(ind_x, max_order)

    # 独立验证集损失
    ind_loss = np.zeros(lambdas.shape)
    # 训练损失
    train_loss = np.zeros(lambdas.shape)

    for idx in range(len(lambdas)):
        theta = normal_equation(poly_x, y, lambdas[idx])
        # 预测
        train_pred = np.dot(poly_x, theta)
        train_loss[idx] = np.mean(np.square(train_pred - y))
        ind_pred = np.dot(poly_ind_x, theta)
        ind_loss[idx] = np.mean(np.square(ind_pred - ind_y))

    return train_loss, ind_loss, lambdas


def main():
    np.random.seed(2)
    # 加载数据
    x, y, ind_x, ind_y = generate_data()
    # 训练
    train_loss, ind_loss, lambdas = bv_train(x, y, ind_x, ind_y)
    # 绘制损失曲线
    plot_loss(train_loss, ind_loss, lambdas)


if __name__ == "__main__":
    main()
