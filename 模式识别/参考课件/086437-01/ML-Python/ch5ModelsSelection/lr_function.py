# -*- coding: utf-8 -*-
"""
逻辑回归函数，使用多项式回归
这里只测试二元分类

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def generate_data(n):
    """
    生成数据集
    输入参数：
        n：样本数
    返回：
        data：n行3列数据集。前两列为数据，第三列为标签
    """
    np.random.seed(1)
    data = 10 * np.random.rand(n, 3) - 5
    for i in range(n):
        if np.square(data[i, 0]) / np.square(3.5) + np.square(data[i, 1]) / np.square(3) > 1:
            data[i, 2] = 1
        else:
            data[i, 2] = 0
        # 将边界上的数据随机加些噪声
        if np.square(np.square(data[i, 0]) / np.square(3.5) + np.square(data[i, 1]) / np.square(3) - 1) < 0.08:
            if np.random.rand(1) > 0.5:
                data[i, 2] = 1
            else:
                data[i, 2] = 0

    return data


def sigmoid(z):
    """ S型激活函数 """
    g = 1 / (1 + np.exp(-z))
    return g


def poly_features(x1, x2):
    """
    将两个特征转换为多项式特征
    例如，将x1, x2转换为1, x1, x2, x1.^2, x2.^2, x1*x2, x1*x2.^2...
    输入
        x1，x2：两个特征都是N行1列
    输出
        x_data：转换后的多项式特征
    """
    degree = 5  # 阶次
    x_data = np.ones((len(x1), 1))  # 截距项
    for i in range(1, degree + 1):
        for j in range(i + 1):
            x_data = np.column_stack((x_data, np.multiply(np.power(x1, (i - j)), np.power(x2, j))))

    return x_data


def compute_cost(x, y, theta, my_lambda):
    """
    计算逻辑回归的代价
    使用theta作为逻辑回归的参数，计算代价J
    输入参数
        x：输入，y：输出，theta：参数，my_lambda：正则化参数
    输出参数
        j_value：计算的J值， grad：梯度
    """
    n = len(y)  # 训练样本数

    temp_theta = theta.copy()
    temp_theta[0, 0] = 0  # 正则化不包括截距项

    h = sigmoid(np.dot(x, theta))
    eps = 1e-6  # 避免数值计算问题
    j_value = -1 / n * np.add(np.dot(y.T, np.log(h + eps)), np.dot((1 - y).T, np.log(1 - h + eps))) + \
              my_lambda / 2 * np.dot(temp_theta.T, temp_theta)
    grad = 1 / n * np.dot(x.T, np.subtract(h, y)) + my_lambda * temp_theta
    return j_value, grad


def gradient_descent(x, y, theta, alpha, iters, my_lambda):
    """
    梯度下降函数，找到合适的参数
    输入参数
        x：输入，y：输出，theta：参数，alpha：学习率，iters：迭代次数，my_lambda：正则化参数
    输出参数
        theta：学习到的参数，j_history：迭代计算的J值历史
    """
    # 初始化
    n = len(y)  # 训练样本数
    j_history = np.zeros((iters,))

    for it in range(iters):
        # 保存代价J
        j_history[it], grad = compute_cost(x, y, theta, my_lambda)
        # 更新theta参数
        theta = theta - alpha / n * grad

    return theta, j_history


def predict(theta, new_x):
    """
    使用学习到的逻辑回归参数theta来预测新样本new_x的标签
    输入参数：
        theta：逻辑回归参数，new_x：新样本集
    返回：
        p：预测为正例的概率，为0~1范围
    """
    p = sigmoid(np.dot(new_x, theta))
    return p


def lr_classifier():
    """
    使用逻辑回归方法预测随机生成的数据集
    返回：
        y ： 真实标签，pred_y： 预测标签，为0~1范围实数，表示标签为1的可能性
    """
    # 梯度下降法的设置
    alpha = 0.1  # 学习率
    iterations = 10000  # 迭代次数

    # 正则化参数lambda
    my_lambda = 0.001

    np.random.seed(1234)
    n = 250     # 样本数
    data = generate_data(n)

    # 随机置乱
    np.random.shuffle(data)
    x = data[:, :2]     # 前两列为属性x
    y = data[:, 2].reshape(-1, 1)      # 第3列为目标属性y
    pred_y = np.zeros_like(y)

    # 添加截距项和多项式项
    x_data = poly_features(x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1))

    d = len(x_data[0])

    k = 10  # K折交叉验证
    sizes = np.tile(np.floor(n / k), k)  # sizes为一维向量，每个单元对应每折的样本数
    sizes[-1] = sizes[-1] + n - np.sum(sizes)  # 如果不能均分，将多出数据放至最后一个单元
    c_sizes = np.cumsum(sizes)          # c_sizes为累积和
    c_sizes = np.insert(c_sizes, 0, 0).astype(int)

    for fold in range(k):
        # 划分数据
        # fold_x仅包含一折数据，为测试折
        # train_x包含其他折数据，为训练折
        fold_x = x_data[c_sizes[fold]: c_sizes[fold + 1]]
        # fold_y = y[c_sizes[fold]: c_sizes[fold + 1]]
        if fold == 0:
            train_x = x_data[c_sizes[fold + 1]:]
            train_y = y[c_sizes[fold + 1]:]
        elif fold == k - 1:
            train_x = x_data[: c_sizes[fold]]
            train_y = y[: c_sizes[fold]]
        else:
            train_x = np.row_stack((x_data[: c_sizes[fold]], x_data[c_sizes[fold + 1]:]))
            train_y = np.row_stack((y[: c_sizes[fold]], y[c_sizes[fold + 1]:]))

        # 初始化theta
        theta = np.zeros((d, 1))
        theta, _ = gradient_descent(train_x, train_y, theta, alpha, iterations, my_lambda)
        # 预测
        pred_y[c_sizes[fold]: c_sizes[fold + 1]] = predict(theta, fold_x)

    return y, pred_y
