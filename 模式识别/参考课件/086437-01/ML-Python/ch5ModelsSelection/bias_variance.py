# -*- coding: utf-8 -*-
"""
诊断是偏差问题还是方差问题

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
    n = 10  # 样本数
    # 生成0到1之间的x
    x = np.linspace(0, 1, n).reshape((-1, 1))
    noise = 0.25          # 噪声强度
    # 目标
    y = np.sin(2 * np.pi * x) + noise * np.random.randn(len(x), 1)
    # 独立验证集
    ind_x = np.linspace(0, 1, 100).reshape((-1, 1))
    ind_y = np.sin(2 * np.pi * ind_x)       # 不加随机噪声
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


def plot_loss(train_loss, ind_loss):
    """
    绘制训练误差和验证误差曲线
    """
    plt.figure()
    plot_x = np.linspace(1, len(ind_loss), len(ind_loss))
    plt.plot(plot_x, train_loss, 'b--', label=u'训练误差')
    plt.plot(plot_x, ind_loss, 'g-', label=u'验证误差')
    plt.xlabel(u'模型阶次')
    plt.ylabel(u'误差')
    plt.title(u'训练误差和验证误差')
    plt.legend()
    plt.show()


def bv_train(x, y, ind_x, ind_y):
    """
    偏差方差训练
    """
    max_order = 9  # 最多为9阶多项式

    # 独立验证集误差
    ind_loss = np.zeros(max_order)
    # 训练误差
    train_loss = np.zeros(max_order)

    # 按模型阶次顺序运行训练和验证
    for order in range(max_order):
        # 构建训练集和测试集矩阵
        poly_x = poly_features(x, order + 1)
        poly_ind_x = poly_features(ind_x, order + 1)

        theta = normal_equation(poly_x, y)
        # 预测
        train_pred = np.dot(poly_x, theta)
        train_loss[order] = np.mean(np.square(train_pred - y))
        ind_pred = np.dot(poly_ind_x, theta)
        ind_loss[order] = np.mean(np.square(ind_pred - ind_y))

    return train_loss, ind_loss


def main():
    np.random.seed(2)
    # 加载数据
    x, y, ind_x, ind_y = generate_data()
    # 训练
    train_loss, ind_loss = bv_train(x, y, ind_x, ind_y)
    # 绘制误差曲线
    plot_loss(train_loss, ind_loss)


if __name__ == "__main__":
    main()
