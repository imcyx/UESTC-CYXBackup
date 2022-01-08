# -*- coding: utf-8 -*-
"""
逻辑回归（Logistic Regression）分类
这里只测试二元分类，使用scipy.optimize.minimize

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def read_csv(file):
    """
    读取CSV文件，允许有多个属性，文件最后一列为目标属性
    输入参数：
        file：文件名
    返回：
        x_data：属性，y：目标属性
    """
    with open(file, encoding="utf-8") as fr:
        fr.readline()  # 跳过标题行
        content = fr.readlines()
        x_data = [f.split(",")[: -1] for f in content]
        y = [f.split(",")[-1].strip("\n") for f in content]
    return x_data, y


def sigmoid(z):
    """ S型激活函数 """
    g = 1 / (1 + np.exp(-z))
    return g


def predict(theta, new_x):
    """
    使用学习到的逻辑回归参数theta来预测新样本new_x的标签
    """
    p = sigmoid(np.dot(new_x, theta))
    return p


def cost(theta, x, y):
    """
    计算逻辑回归的代价
    使用theta作为逻辑回归的参数，计算代价J
    输入参数
        x：输入，y：输出，theta：参数
    输出参数
        j_value：计算的J值
    """
    n = len(y)  # 训练样本数
    h = sigmoid(np.dot(x, theta))
    j_value = 1 / n * (np.dot(-y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h)))
    return j_value


def gradient(theta, x, y):
    """
    计算线性回归代价的梯度
    使用w作为线性回归的参数，计算代价J
    输入参数
        x：输入，y：输出，theta：参数
    输出参数
        grad：梯度
    """
    n = len(y)  # 训练样本数
    h = sigmoid(np.dot(x, theta))
    grad = 1 / n * np.dot(x.T, (h - y))
    return grad


def plot_decision_boundary(x, y, theta, j_value):
    """
    绘制决策边界
    输入参数
        x：输入，y：输出，theta：参数，j_value：代价
    输出参数
        无
    """
    plt.figure()
    neg_x = x[np.where(y[:, 0] == 0)]
    pos_x = x[np.where(y[:, 0] == 1)]
    neg = plt.scatter(neg_x[:, 1], neg_x[:, 2], c='r', marker='o')
    pos = plt.scatter(pos_x[:, 1], pos_x[:, 2], c='b', marker='+')
    # 绘制决策边界
    # 只需要两点便可以定义一条直线，选择两个端点
    plot_x = np.array([min(x[:, 1]), max(x[:, 1])])
    # 计算决策边界线，theta0 + theta1*x + theta2*y = 0
    # 已知x，可计算y
    plot_y = np.dot(np.divide(-1, theta[2][0]), (theta[0][0] + np.dot(theta[1][0], plot_x)))
    lr, = plt.plot(plot_x, plot_y, 'g')  # 绘制拟合直线

    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend([neg, pos, lr], ['Versicolor', 'Virginica', u'决策边界'], loc='lower left')
    plt.title(u'代价J=%f' % j_value)
    plt.show()


def main():
    #  加载鸢尾花数据
    file_path = "../data/fisheriris.csv"
    x, _ = read_csv(file_path)

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    # 3、4列分别是petal length和petal width，仅使用这两个属性
    # 仅使用versicolor和virginica两类
    x_data = x_data[50: 150, [2, 3]]
    # 目标versicolor为0，virginica为1
    y_data = np.row_stack((np.zeros((50, 1)), np.ones((50, 1))))

    # 使用minimize优化可以不规范化

    n = len(y_data)  # 样本数

    # 添加一列全1，以扩展x
    x_data = np.column_stack((np.ones((n, 1)), x_data))
    theta = np.zeros(3)  # 参数初始值

    res = opt.minimize(fun=cost, x0=theta, args=(x_data, y_data))
    if not res.success:
        print("优化错误！")
        import sys
        sys.exit(1)
    j_value = res.fun
    theta = res.x.reshape(-1, 1)
    plot_decision_boundary(x_data, y_data, theta, j_value)

    # 预测并计算在训练集上的正确率
    new_x = np.array([1, 5.6, 2.2]).reshape(1, -1)
    prob = np.squeeze(predict(theta, new_x))
    print('新样本的Petal length和Petal width分别为5.6和2.2，预测为virginica的概率为： %f\n\n' % prob)

    # 计算在训练集上的分类正确率
    y_hat = predict(theta, x_data) >= 0.5
    print('训练集上的分类正确率：{:.2%}'.format(np.mean(y_hat == y_data)))


if __name__ == "__main__":
    main()
