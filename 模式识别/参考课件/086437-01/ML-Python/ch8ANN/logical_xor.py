# -*- coding: utf-8 -*-
"""
逻辑异或示例
使用神经元模拟XOR逻辑
利用等式 x1 XOR x2 = ((NOT x1) AND x2) OR (x1 AND (NOT x2))

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


def get_data():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float)  # 两列为属性x1、x2
    y = np.array([0, 1, 1, 0], dtype=np.float)
    return x, y


def sigmoid(z):
    """ S型激活函数 """
    g = 1 / (1 + np.exp(-z))
    return g


def compute_cost(theta, x, y):
    """
    计算逻辑回归的代价
    使用theta作为逻辑回归的参数，计算代价J和梯度
    输入参数
        theta：参数，x：输入，y：输出
    输出参数
        j_value：计算的J值， grad：梯度
    """
    n = len(y)  # 训练样本数
    h = sigmoid(np.dot(x, theta))
    eps = 1e-6      # 避免数值计算问题
    j_value = 1 / n * (np.dot(-y.T, np.log(h + eps)) - np.dot((1 - y).T, np.log(1 - h + eps)))
    grad = 1 / n * np.dot(x.T, (h - y))
    return j_value, grad


def predict(theta, new_x):
    """
    使用学习到的逻辑回归参数theta来预测新样本new_x的标签
    """
    p = sigmoid(np.dot(new_x, theta))
    return p


def predict_xor(x):
    """
    使用神经元模拟XOR逻辑
    """
    theta_and = np.array([-47.21428427, 30.46394204, 30.46394204])      # 逻辑与的参数
    theta_or = np.array([-12.22026859, 26.83899661, 26.83899661])       # 逻辑或的参数
    theta_not = np.array([14.89858169, -28.94841338])                   # 逻辑非的参数

    # 计算 NOT x1
    not_x1 = predict(theta_not, x[:, [0, 1]])
    # 计算 NOT x2
    not_x2 = predict(theta_not, x[:, [0, 2]])
    # 计算 (NOT x1) AND x2
    not_x1_and_x2 = predict(theta_and, np.column_stack((np.ones((len(x), 1)), not_x1, x[:, 2])))
    # 计算 x1 AND (NOT x2)
    x1_and_not_x2 = predict(theta_and, np.column_stack((np.ones((len(x), 1)), x[:, 1], not_x2)))
    # 计算 x1 XOR x2 = ((NOT x1) AND x2) OR (x1 AND (NOT x2))
    x1_xor_x2 = predict(theta_or, np.column_stack((np.ones((len(x), 1)), not_x1_and_x2, x1_and_not_x2)))
    return x1_xor_x2


def plot_decision_boundary(x, y):
    """
    绘制散点图
    输入参数：
        x：特征，y：目标属性
    返回：
        无
    """
    plt.figure()
    neg_x = x[np.where(y == 0)]
    pos_x = x[np.where(y == 1)]
    neg = plt.scatter(neg_x[:, 1], neg_x[:, 2], c='r', marker='o')
    pos = plt.scatter(pos_x[:, 1], pos_x[:, 2], c='b', marker='+')

    # 绘制决策边界
    # 网格范围
    u = np.linspace(min(x[:, 1]), max(x[:, 1]), 150)
    v = np.linspace(min(x[:, 2]), max(x[:, 2]), 150)
    uu, vv = np.meshgrid(u, v)  # 生成网格数据
    z = predict_xor(np.column_stack((np.ones((len(uu.ravel()), 1)), uu.ravel(), vv.ravel())))
    # 保持维度一致
    z = z.reshape(uu.shape)
    # 画图
    plt.contour(uu, vv, z, 0)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend([neg, pos], ['y=0', 'y=1'])
    plt.title(u'逻辑异或')

    plt.show()


def main():
    # 训练数据
    x, y = get_data()

    n = len(y)  # 样本数

    # 加上截距项
    x = np.column_stack((np.ones((n, 1)), x))

    y_hat = predict_xor(x) >= 0.5

    # 计算在训练集上的分类正确率
    print('训练集上的分类正确率:{:.2%}'.format(np.mean(y_hat == y)))

    # 绘图
    plot_decision_boundary(x, y)


if __name__ == "__main__":
    main()
