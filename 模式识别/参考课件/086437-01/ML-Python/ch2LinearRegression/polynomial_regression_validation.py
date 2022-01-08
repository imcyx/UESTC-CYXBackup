# -*- coding: utf-8 -*-
"""
多项式回归拟合历届奥运会自由泳100米数据
划分训练集与验证集，验证

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""
from typing import List

import numpy as np
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
        content: List[str] = fr.readlines()
        x = [f.split(",")[: -1] for f in content]
        y = [float(f.split(",")[-1].strip("\n")) for f in content]
    return x, y


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


def plot_scatter(train_x, train_y, val_x, val_y):
    """
    绘制散点图
    输入参数：
        train_x：训练属性，train_y：训练目标属性，val_x：验证属性，val_y：验证目标属性
    返回：
        无
    """
    plt.scatter(train_x, train_y, c='b', marker='o', s=5, linewidths=2)
    plt.scatter(val_x, val_y, c='r', marker='s', s=5, linewidths=2)
    lim = np.array([min(train_x) - 4, max(val_x) + 4])  # 为上下限留一点空隙
    plt.xlim(lim)  # 设定x坐标轴的范围
    plt.ylim((40, 85))  # 设定y坐标轴的范围
    plt.xlabel(u'奥运会举办年编号')
    plt.ylabel(u'取胜时间（秒）')
    plt.title(u'多项式回归')


def poly_regression(train_x, train_y, val_x, val_y, rank):
    """
    绘制散点图
    输入参数：
        train_x：训练属性，train_y：训练目标属性，rank：多项式阶次
    返回：
        无
    """
    n_train = len(train_y)  # 训练样本数
    n_val = len(val_y)      # 验证样本数
    x_train_data = np.ones((n_train, 1))
    x_val_data = np.ones((n_val, 1))
    for i in range(rank):
        x_train_data = np.column_stack((x_train_data, np.power(train_x, i + 1)))
        x_val_data = np.column_stack((x_val_data, np.power(val_x, i + 1)))

    theta = normal_equation(x_train_data, train_y)

    # 计算并打印训练损失和验证损失
    train_loss = np.squeeze(np.mean(np.power(np.subtract(np.dot(x_train_data, theta), train_y), 2)))
    val_loss = np.squeeze(np.mean(np.power(np.subtract(np.dot(x_val_data, theta), val_y), 2)))
    print(f'模型阶次： {rank}， 训练损失： {train_loss}，验证损失： {val_loss}')

    # 绘制拟合线
    # 显示的数据点。左右留边距2
    p_x = np.arange(min(train_x) - 2, max(val_x) + 2, 0.01).reshape(-1, 1)
    plot_x = np.ones((len(p_x), 1))
    for i in range(rank):
        plot_x = np.column_stack((plot_x, np.power(p_x, i + 1)))
    plt.plot(p_x, np.dot(plot_x, theta).reshape(-1, 1))


def main():
    #    加载奥运会数据
    file_path = "../data/Freestyle100m.csv"
    x, y = read_csv(file_path)

    x = [float(f[0]) for f in x]
    # 1984年前的数据用于训练，之后用于验证
    pos = x.index(1984)
    # 转换为Numpy数组
    x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
    # 将奥运会举办年转换为届，避免数值计算问题
    x -= x[0]
    x /= 4

    val_x = x[pos:]  # 将1984年之后的数据用于验证集
    val_y = y[pos:]
    train_x = x[: pos]
    train_y = y[: pos]

    plt.figure()
    # 绘图
    plot_scatter(train_x, train_y, val_x, val_y)

    # 拟合不同模型并绘制结果
    orders = [1, 2, 4, 8]  # 要拟合这些阶次的模型
    for i in orders:
        poly_regression(train_x, train_y, val_x, val_y, i)

    # 显示绘图结果
    plt.legend(['线性模型', '2次模型', '4次模型', '8次模型', '训练', '验证'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
