# -*- coding: utf-8 -*-
"""
单变量线性回归
使用梯度下降技术，线性拟合奥运会自由泳100米记录

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['simhei']
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


def compute_cost(x, y, w):
    """
    计算线性回归的代价
    使用w作为线性回归的参数，计算代价J
    输入参数
        x：输入，y：输出，w：截距和斜率参数
    输出参数
        j：计算的J值
    """
    n = len(y)  # 训练样本数
    j = 1 / (2 * n) * np.sum(np.power(np.subtract(np.dot(x, w), y), 2))
    return j


def gradient_descent(x, y, w, alpha, iters):
    """
    梯度下降函数，找到合适的参数
    输入参数
        x：输入，y：输出，w：截距和斜率参数，alpha：学习率，iters：迭代次数
    输出参数
        w：学习到的截距和斜率参数，j_history：迭代计算的J值历史
    """
    # 初始化
    n = len(y)  # 训练样本数
    j_history = np.zeros((iters,))

    it: int
    for it in range(iters):
        # 要求同时更新w参数，因此设置两个临时变量temp0和temp1
        temp0 = w[0] - alpha / n * (np.dot(x[:, 0].T, np.dot(x, w) - y))
        temp1 = w[1] - alpha / n * (np.dot(x[:, 1].T, np.dot(x, w) - y))
        w[0] = temp0
        w[1] = temp1

        # 保存代价J
        j_history[it] = compute_cost(x, y, w)
    return w, j_history


def plot_scatter(x, y, w0, w1):
    """
    绘制散点图
    输入参数：
        x：一个属性，y：目标属性，w0：截距，w1：斜率
    返回：
        无
    """
    plt.figure()
    plt.scatter(x, y, c='b', marker='o', s=5, linewidths=2)
    lim = np.array([min(x) - 4, max(x) + 4])  # 为上下限留一点空隙
    plt.xlim(lim)  # 设定x坐标轴的范围
    plt.plot(lim, w0 + w1 * lim, 'r')  # 绘制拟合直线
    plt.xlabel(u'奥运会年')
    plt.ylabel(u'取胜时间（秒）')

    plt.show()


def plot_contour(x, y, w):
    """
    可视化J(w)
    输入参数：
        x：一个属性，y：目标属性，w：参数，j_history：J历史
    返回：
        无
    """
    # 代价函数J的绘图范围
    w0 = np.linspace(-30, 170, 100)
    w1 = np.linspace(-2, 2, 100)

    # 初始化J的值
    j_value = np.zeros((len(w0), len(w1)))

    # 填充J的值
    for i in range(len(w0)):
        for j in range(len(w1)):
            t = np.array([w0[i], w1[j]]).reshape((2, 1))
            j_value[i, j] = compute_cost(x, y, t)

    j_value = j_value.T

    # 绘制曲面图
    from mpl_toolkits import mplot3d    # 必须要这一句，否则出错
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(w0, w1, j_value, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
    ax.set_zlabel('j_value')
    plt.show()

    # 绘制等值线图
    plt.figure()
    # 绘制代价J的等值线图
    plt.contour(w0, w1, j_value, np.logspace(-2, 3, 20))
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.plot(w[0], w[1], 'rx', markersize=10, linewidth=2)
    plt.show()


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
    # 加载奥运会数据
    file_path = "../data/Freestyle100m.csv"
    x, y = read_csv(file_path)

    x = [float(f[0]) for f in x]
    x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)  # 转换为Numpy数组
    n = len(y)  # 样本数

    # 梯度下降
    # 为方便数值运算，将原来的举办年减去第一届奥运会年（1896），并添加一列全1，以扩展x
    x = np.column_stack((np.ones((n, 1)), x - 1896))
    w = np.zeros((2, 1))  # 参数初始值

    # 梯度下降法的超参数设置
    iterations = 50000  # 迭代次数
    alpha = 0.00034  # 学习率

    # 调用梯度下降函数
    w, j_history = gradient_descent(x, y, w, alpha, iterations)

    # 打印找到的参数
    print('梯度下降找到的w0和w1：%f %f \n' % (w[0], w[1]))

    # 绘图
    plot_scatter(x[:, 1], y, w[0], w[1])
    plot_contour(x, y, w)
    plot_jhistory(j_history)


if __name__ == "__main__":
    main()
