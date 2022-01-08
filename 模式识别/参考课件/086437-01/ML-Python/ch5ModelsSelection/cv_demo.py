# -*- coding: utf-8 -*-
"""
演示模型选择的交叉验证

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
    n = 100  # 样本数
    # 生成-5到5之间的x
    x = 10 * np.random.rand(n, 1) - 5
    w = [2, -1]  # 真实的参数
    noise = 20
    # 目标
    y = w[1] * np.square(x) + w[0] * x + 100 + noise * np.random.randn(len(x), 1)
    # 独立测试集
    ind_x = np.linspace(-5, 5, n).reshape((-1, 1))
    ind_y = w[1] * np.square(ind_x) + w[0] * ind_x + 100 + noise * np.random.randn(len(ind_x), 1)
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


def plot_loss(cv_cost, ind_cost, train_cost):
    """
    绘制交叉验证误差、训练误差和独立测试集误差曲线
    """
    plt.figure()
    plot_x = np.linspace(1, len(cv_cost), len(cv_cost))
    plt.subplot(1, 3, 1)
    plt.plot(plot_x, cv_cost)
    plt.xlabel(u'模型阶次')
    plt.ylabel(u'误差')
    plt.xticks(np.arange(0, len(cv_cost), 2))
    plt.title(u'交叉验证误差')
    plt.subplot(1, 3, 2)
    plt.plot(plot_x, train_cost)
    plt.xlabel(u'模型阶次')
    plt.ylabel(u'误差')
    plt.xticks(np.arange(0, len(cv_cost), 2))
    plt.title(u'训练误差')
    plt.subplot(1, 3, 3)
    plt.plot(plot_x, ind_cost)
    plt.xlabel(u'模型阶次')
    plt.ylabel(u'误差')
    plt.xticks(np.arange(0, len(cv_cost), 2))
    plt.title(u'独立测试集误差')
    plt.show()


def cv_train(x, y, ind_x, ind_y):
    """
    交叉验证训练
    输入
        x，y：特征矩阵和目标属性，ind_x，ind_y：独立验证集特的征矩阵和目标属性
    输出
        cv_loss：交叉验证误差，ind_loss：独立验证误差，train_loss：训练误差
    """
    # 按模型顺序运行交叉验证
    max_order = 7  # 最多为7阶多项式
    k = 10  # K折交叉验证
    n = len(y)
    sizes = np.tile(np.floor(n / k), k)  # sizes为一维向量，每个单元对应每折的样本数
    sizes[-1] = sizes[-1] + n - np.sum(sizes)  # 如果不能均分，将多出数据放至最后一个单元
    c_sizes = np.cumsum(sizes)  # c_sizes为累积和
    c_sizes = np.insert(c_sizes, 0, 0).astype(int)

    # 注意一般在交叉验证前需要打乱数据次序。这里的x是随机产生的，因此不必要做这一步

    # 交叉验证误差
    cv_loss = np.zeros((max_order, k))
    # 独立测试集误差
    ind_loss = np.zeros((max_order, k))
    # 训练误差
    train_loss = np.zeros((max_order, k))

    for order in range(max_order):
        # 构建训练集和测试集矩阵
        poly_x = poly_features(x, order + 1)
        poly_ind_x = poly_features(ind_x, order + 1)

        for fold in range(k):
            # 划分数据
            # fold_x仅包含一折数据，为测试折
            # train_x包含其他折数据，为训练折
            fold_x = poly_x[c_sizes[fold]: c_sizes[fold + 1]]
            fold_y = y[c_sizes[fold]: c_sizes[fold + 1]]
            if fold == 0:
                train_x = poly_x[c_sizes[fold + 1]:]
                train_y = y[c_sizes[fold + 1]:]
            elif fold == k - 1:
                train_x = poly_x[: c_sizes[fold]]
                train_y = y[: c_sizes[fold]]
            else:
                train_x = np.row_stack((poly_x[: c_sizes[fold]], poly_x[c_sizes[fold + 1]:]))
                train_y = np.row_stack((y[: c_sizes[fold]], y[c_sizes[fold + 1]:]))

            theta = normal_equation(train_x, train_y)
            # 预测
            fold_pred = np.dot(fold_x, theta)
            cv_loss[order, fold] = np.mean(np.square(fold_pred - fold_y))
            ind_pred = np.dot(poly_ind_x, theta)
            ind_loss[order, fold] = np.mean(np.square(ind_pred - ind_y))
            train_pred = np.dot(train_x, theta)
            train_loss[order, fold] = np.mean(np.square(train_pred - train_y))

    return cv_loss.mean(axis=1), ind_loss.mean(axis=1), train_loss.mean(axis=1)


def main():
    np.random.seed(2)
    # 加载数据
    x, y, ind_x, ind_y = generate_data()
    cv_loss, ind_loss, train_loss = cv_train(x, y, ind_x, ind_y)
    plot_loss(cv_loss, ind_loss, train_loss)


if __name__ == "__main__":
    main()
