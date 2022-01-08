# -*- coding: utf-8 -*-
"""
本例展示线性回归不适合解决分类问题

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


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


def plot_classifier(x, y):
    """
    绘制假想的癌症数据
    输入
        x：特征矩阵，y：目标属性
    输出
        无
    """
    n = len(y)
    neg_x = x[np.where(y == 0)]
    neg_y = y[np.where(y == 0)]
    pos_x = x[np.where(y == 1)]
    pos_y = y[np.where(y == 1)]
    # 添加一列全1，以扩展x
    x_data = np.column_stack((np.ones((n, 1)), x))
    # 用正规方程计算theta参数
    theta = normal_equation(x_data, y)

    plt.figure(figsize=(8, 2))
    neg = plt.scatter(neg_x, neg_y, c='r', marker='o')
    pos = plt.scatter(pos_x, pos_y, c='b', marker='x')
    x_lim = np.array([min(x) - 2, max(x) + 2])  # 为上下限留一点空隙
    plt.xlim(x_lim)  # 设定x坐标轴的范围
    plt.ylim((0, 1.2))  # 设定x坐标轴的范围
    lr, = plt.plot(x_lim, theta[0] + theta[1] * x_lim, 'g')  # 绘制拟合直线
    plt.legend([neg, pos, lr], ['负例', '正例', '线性回归'], loc='lower right')
    db = (0.5 - theta[0]) / theta[1]    # 决策边界
    plt.plot([-1, db], [0.5, 0.5], '--')
    plt.plot([db, db], [0, 1.2], 'm--')
    plt.xticks([])
    plt.yticks(np.arange(0, 1.2, 0.5))
    plt.xlabel(u'肿瘤大小')
    plt.ylabel(u'恶性良性')

    plt.show()


def main():
    # 假想的癌症数据
    data = [[1, 0],
            [2, 0],
            [3, 0],
            [4, 1],
            [5, 1],
            [6, 1]]
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    # 绘制线性回归决策面
    plot_classifier(x, y)

    # 改变一个样本，然后重复用线性回归
    # 可以看到线性回归解决分类问题的缺陷
    x[5] = 20
    plot_classifier(x, y)


if __name__ == "__main__":
    main()
