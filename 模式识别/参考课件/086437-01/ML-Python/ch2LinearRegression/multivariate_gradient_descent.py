# -*- coding: utf-8 -*-
"""
多变量线性回归
使用梯度下降技术，线性拟合UCI数据集——Relative CPU Performance Data

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

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
        content = fr.readlines()
        x = [f.split(",")[: -1] for f in content]
        y = [float(f.split(",")[-1].strip("\n")) for f in content]
    return x, y


def feature_normalize(x):
    """
    特征规范化
    将特征规范化为0均值1方差
    输入:
        x：要规范化的特征
    输出:
        normalized_x：已规范化的特征，mu：均值，sigma：方差
    """
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    normalized_x = (x - mu) / sigma
    return normalized_x, mu, sigma


def compute_cost(x, y, theta):
    """
    计算线性回归的代价
    使用w作为线性回归的参数，计算代价J
    输入参数
        x：输入，y：输出，theta：参数
    输出参数
        j_value：计算的J值
    """
    n = len(y)  # 训练样本数
    j_value = 1 / (2 * n) * np.sum(np.power(np.subtract(np.dot(x, theta), y), 2))
    return j_value


def gradient_descent(x, y, theta, alpha, iters):
    """
    梯度下降函数，找到合适的参数
    输入参数
        x：输入，y：输出，theta：参数，alpha：学习率，iters：迭代次数
    输出参数
        theta：学习到的参数，j_history：迭代计算的J值历史
    """
    # 初始化
    n = len(y)  # 训练样本数
    j_history = np.zeros((iters,))

    for it in range(iters):
        # 用矩阵运算同时更新theta参数
        theta = theta - alpha / n * (np.dot(x.T, np.dot(x, theta) - y))
        # 保存代价J
        j_history[it] = compute_cost(x, y, theta)
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
    # 梯度下降法的超参数设置
    iterations = 300  # 迭代次数
    alpha = 0.01  # 学习率

    # 加载CPU数据
    file_path = "../data/machine.csv"
    x, y = read_csv(file_path)

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    y_data = np.array(y).reshape(-1, 1)

    # 规范化特征，使其变为0均值1方差
    x_data, mu, sigma = feature_normalize(x_data)
    n = len(y_data)  # 样本数

    # 梯度下降
    # 添加一列全1，以扩展x
    x_data = np.column_stack((np.ones((n, 1)), x_data))
    theta = np.zeros((7, 1))  # 参数初始值

    # 调用梯度下降函数
    theta, j_history = gradient_descent(x_data, y_data, theta, alpha, iterations)

    # 打印找到的参数
    print(f'梯度下降结果，Theta参数：\n {theta}')

    # 计算误差
    estimate_y = np.dot(x_data, theta)
    err = y_data - estimate_y
    rmse = np.squeeze(np.sqrt(np.dot(err.T, err) / n))
    print(f'\nRMSE：  {rmse} \n')

    # 绘图
    plot_jhistory(j_history)


if __name__ == "__main__":
    main()
