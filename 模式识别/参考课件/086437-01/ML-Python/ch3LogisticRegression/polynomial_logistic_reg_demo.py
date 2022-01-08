# -*- coding: utf-8 -*-
"""
逻辑回归示例，使用多项式回归
这里只测试二元分类

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 使用动画
use_animation = False


def generate_data(n):
    """
    生成数据集
    输入
        n：样本数
    输出
        data：n行3列数据集。前两列为数据，第三列为标签
    """
    np.random.seed(1)
    data = np.random.rand(n, 3) * 10 - 5
    for i in range(n):
        if data[i, 0] ** 2 / 3.5 ** 2 + data[i, 1] ** 2 / 3 ** 2 > 1:
            data[i, 2] = 1
        else:
            data[i, 2] = 0

        # 将边界上的数据随机加些噪声
        if data[i, 0] ** 2 / 3.5 ** 2 + data[i, 1] ** 2 / 3 ** 2 < 0.08:
            if np.random.rand(1) > 0.5:
                data[i, 2] = 1
            else:
                data[i, 2] = 0
    return data


def plot_data(x, y):
    """ 绘制二维数据集散点图 """
    plt.figure()
    neg_x = x[np.where(y[:, 0] == 0)]
    pos_x = x[np.where(y[:, 0] == 1)]
    neg = plt.scatter(neg_x[:, 0], neg_x[:, 1], c='b', marker='o')
    pos = plt.scatter(pos_x[:, 0], pos_x[:, 1], c='r', marker='+')

    # 坐标
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 图例
    plt.legend([neg, pos], ['负例', '正例'], loc='lower right')
    plt.show()


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


def poly_features(x1, x2):
    """
    将两个特征转换为多项式特征
    例如，将x1, x2转换为1, x1, x2, x1.^2, x2.^2, x1*x2, x1*x2.^2...
    输入
        x1，x2：两个特征都是N行1列
    输出
        x_data：转换后的多项式特征
    """
    degree = 8  # 阶次
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
    eps = 1e-6
    j_value = -1 / n * np.add(np.dot(y.T, np.log(h + eps)), np.dot((1 - y).T, np.log(1 - h + eps))) + \
              my_lambda / 2 * np.dot(temp_theta.T, temp_theta)
    grad = 1 / n * np.dot(x.T, np.subtract(h, y)) + my_lambda * temp_theta
    return j_value, grad


def plot_decision_boundary(x, y, theta, j_value):
    """
    绘制决策边界
    输入参数
        x：输入，y：输出，theta：参数，j_value：代价
    输出参数
        无
    """
    neg_x = x[np.where(y[:, 0] == 0)]
    pos_x = x[np.where(y[:, 0] == 1)]
    plt.clf()
    neg = plt.scatter(neg_x[:, 1], neg_x[:, 2], c='r', marker='o')
    pos = plt.scatter(pos_x[:, 1], pos_x[:, 2], c='b', marker='+')
    # 绘制决策边界
    # 网格范围
    u = np.linspace(min(x[:, 1]), max(x[:, 1]), 150)
    v = np.linspace(min(x[:, 2]), max(x[:, 2]), 150)
    uu, vv = np.meshgrid(u, v)  # 生成网格数据
    z = np.dot(poly_features(uu.ravel(), vv.ravel()), theta)
    # 保持维度一致
    z = z.reshape(uu.shape)
    # 画图
    plt.contour(uu, vv, z, 0)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend([neg, pos], [u'负例', u'正例'], loc='lower right')
    plt.title(u'代价J=%f' % j_value)
    plt.pause(0.1)


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

    plt.figure()

    # 打开交互模式
    plt.ion()
    for it in range(iters):
        # 保存代价J
        j_history[it], grad = compute_cost(x, y, theta, my_lambda)
        # 更新theta参数
        theta = theta - alpha / n * grad
        # 使用动画
        if (it % 20 == 0) and use_animation:
            plot_decision_boundary(x, y, theta, j_history[it])
    # 不使用动画
    if not use_animation:
        plot_decision_boundary(x, y, theta, j_history[-1])
    # 关闭交互模式
    plt.ioff()
    plt.show()

    return theta, j_history


def plot_jhistory(j_history):
    """
    绘制代价J的下降曲线
    输入参数：
        Jhistory：J历史
    返回：
        无
    """
    plt.figure()
    plt.plot(j_history, 'r-')  # 绘制J历史曲线
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'代价J')

    plt.show()


def main():
    # 梯度下降法的设置
    alpha = 0.005  # 学习率
    iterations = 1000  # 迭代次数
    # 正则化参数lambda
    # 可以尝试用不同的lambda值，了解该参数如何影响决策边界
    my_lambda = 10

    #  随机生成数据
    data = generate_data(250)
    x = data[:, [0, 1]]  # 前两列为属性x
    y = data[:, 2].reshape(-1, 1)  # 第3列为目标属性y

    plot_data(x, y)

    # 添加截距项和多项式项
    x_data = poly_features(x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1))

    # 初始化Theta参数
    theta = np.zeros((len(x_data[0]), 1))

    # 梯度下降
    theta, j_history = gradient_descent(x_data, y, theta, alpha, iterations, my_lambda)

    plot_jhistory(j_history)

    # 计算在训练集上的分类正确率
    y_hat = predict(theta, x_data) >= 0.5

    print('训练集上的分类正确率：{:.2%}'.format(np.mean(y_hat == y)))


if __name__ == "__main__":
    main()
