# -*- coding: utf-8 -*-
"""
逻辑非示例
使用神经元模拟NOT逻辑

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


def get_data():
    x = np.array([0, 1], dtype=np.float)  # 一列为属性x1
    y = np.array([1, 0], dtype=np.float)
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
    j_value = 1 / n * (np.dot(-y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h)))
    grad = 1 / n * np.dot(x.T, (h - y))
    return j_value, grad


def predict(theta, new_x):
    """
    使用学习到的逻辑回归参数theta来预测新样本new_x的标签
    """
    p = sigmoid(np.dot(new_x, theta))
    return p


def plot_decision_boundary(x, y, theta):
    """
    绘制散点图
    输入参数：
        x：特征，y：目标属性，theta：逻辑回归参数
    返回：
        无
    """
    plt.figure()
    neg_x = x[np.where(y == 0)]
    pos_x = x[np.where(y == 1)]
    neg = plt.scatter(neg_x[:, 1], 0, c='r', marker='o')
    pos = plt.scatter(pos_x[:, 1], 0, c='b', marker='+')
    # 绘制决策边界
    # 只需要两点便可以定义一条直线，选择两个端点
    plot_y = np.array([-1, 1])
    # 计算决策边界线，theta0 + theta1*x = 0
    # 可计算常数x
    plot_x = [-theta[0] / theta[1], -theta[0] / theta[1]]
    db, = plt.plot(plot_x, plot_y, 'g')  # 绘制拟合直线

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.yticks([])
    plt.legend([neg, pos, db], ['y=0', 'y=1', u'决策边界'], loc='upper right')
    plt.title(u'逻辑非')

    plt.show()


def main():
    # 训练数据
    x, y = get_data()

    n = len(y)  # 样本数

    # 加上截距项
    x = np.column_stack((np.ones((n, 1)), x.reshape(-1, 1)))
    init_w = np.zeros(len(x[0]))  # 参数初始值

    # 优化
    options = {'maxiter': 100}
    res = optimize.minimize(lambda p: compute_cost(p, x, y), init_w, jac=True, method='TNC',
                            options=options)
    w = res.x
    # 打印找到的参数
    print(f'优化函数找到的w：{w} \n')

    # 计算在训练集上的分类正确率
    y_hat = predict(w, x) >= 0.5
    print('训练集上的分类正确率：{:.2%}'.format(np.mean(y_hat == y)))

    # 绘图
    plot_decision_boundary(x, y, w)


if __name__ == "__main__":
    main()
