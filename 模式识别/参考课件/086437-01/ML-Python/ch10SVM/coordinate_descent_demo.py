# -*- coding: utf-8 -*-
"""
单变量线性回归
使用坐标下降技术，线性拟合奥运会自由泳100米记录

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from operator import mod
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def read_csv(file):
    """
    读取CSV文件，允许有多个属性，文件最后一列为目标属性
    输入参数：
        file：文件名
    返回：
        x：属性，y：目标属性
    """
    with open(file, encoding="utf-8") as fr:
        content = fr.readlines()
        x = [f.split(",")[: -1] for f in content]
        y = [float(f.split(",")[-1].strip("\n")) for f in content]
    return x, y


def compute_cost(x, y, alpha):
    """
    计算线性回归的代价
    使用w作为线性回归的参数，计算代价J
    输入参数
        x：输入，y：输出，alpha：截距和斜率参数
    输出参数
        j：计算的j值
    """
    n = len(y)      # 训练样本数
    j = 1 / (2 * n) * np.sum(np.square(np.dot(x, alpha) - y))
    return j


def coordinate_descent(x, y, alpha, iters):
    """
    坐标下降函数，寻找合适的参数
    输入参数
        x：输入，y：输出，alpha：截距和斜率参数，iters：最大迭代次数
    输出参数
        alpha：学习到的截距和斜率参数，alpha_history：迭代计算的alpha值历史
    """
    # 初始化
    n = len(y)       # 训练样本数
    alpha_history = np.zeros((2, iters + 1))
    alpha_history[:, 0] = alpha

    for iter in range(1, iters + 1):
        if mod(iter, 2) == 1:
            # 1、 先固定alpha0，求最优的alpha1
            alpha[1] = np.dot(x[:, 1].T, (y - alpha[0])) / np.dot(x[:, 1].T, x[:, 1])
        else:
            # 2、 然后固定alpha1，求最优的alpha0
            alpha[0] = np.sum(y - alpha[1] * x[:, 1]) / n

        # 保存alpha值历史
        alpha_history[:, iter] = alpha

    return alpha, alpha_history


def plot_alpha_history(alpha, alpha_history, x, y):
    """
    可视化 J(alpha0, alpha1)
    输入参数：
        alpha：截距和斜率参数，alpha_history：alpha历史值，x：输入，y：输出
    返回：
        无
    """
    # 代价函数J的绘图范围
    alpha0 = np.linspace(-30, 170, 100)
    alpha1 = np.linspace(-2, 2, 100)

    # 初始化J的值
    j_value = np.zeros((len(alpha0), len(alpha1)))

    # 填充J的值
    for i in range(len(alpha0)):
        for j in range(len(alpha1)):
            j_value[i, j] = compute_cost(x, y, [alpha0[i], alpha1[j]])

    j_value = j_value.T

    # 绘制等值线图
    plt.figure()
    # 绘制代价J的等值线图
    plt.contour(alpha0, alpha1, j_value, np.logspace(-2, 3, 20))
    plt.xlabel(r'$\alpha$0')
    plt.ylabel(r'$\alpha$1')
    plt.plot(alpha[0], alpha[1], 'rx', markersize=10, linewidth=2)
    plt.plot(alpha_history[0, :], alpha_history[1, :])

    plt.show()


def main():
    # 最大迭代次数
    iters = 50

    # 加载奥运会数据
    file_path = "../data/Freestyle100m.csv"
    x, y = read_csv(file_path)
    x = [float(f[0]) for f in x]
    x, y = np.array(x), np.array(y)  # 转换为Numpy数组
    n = len(y)  # 样本数

    # 坐标下降
    # 为方便数值运算，将原来的举办年减去第一届奥运会年（1896），并添加一列全1，以扩展x
    x = np.column_stack((np.ones((n, 1)), x - 1896))
    alpha = np.array([120, 1.5])        # 参数初始值

    # 调用坐标下降函数
    alpha, alpha_history = coordinate_descent(x, y, alpha, iters)

    # 打印找到的参数
    print('坐标下降法找到的alpha0和alpha1：%f %f \n' % (alpha[0], alpha[1]))

    plot_alpha_history(alpha, alpha_history, x, y)


if __name__ == "__main__":
    main()
