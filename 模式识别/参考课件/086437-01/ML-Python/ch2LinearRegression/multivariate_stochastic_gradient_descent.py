# -*- coding: utf-8 -*-
"""
多变量线性回归
使用随机梯度下降技术，线性拟合UCI数据集——Relative CPU Performance Data

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""
from typing import Optional, Any

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
    normalized_x: Optional[Any] = (x - mu) / sigma
    return normalized_x, mu, sigma


def compute_cost(x, y, theta, idx):
    """
    计算线性回归的代价
    使用w作为线性回归的参数，计算代价J
    输入参数
        x：输入，y：输出，theta：参数，idx：样本索引
    输出参数
        cost：计算的代价
    """
    xi = x[idx].reshape(1, -1)
    yi = y[idx].reshape(1, -1)
    cost = 1 / 2 * np.squeeze(np.power(np.subtract(np.dot(xi, theta), yi), 2))
    return cost


def stochastic_gradient_descent(x, y, theta, alpha, iters):
    """
    随机梯度下降函数，找到合适的参数
    输入参数
        x：输入，y：输出，theta：参数，alpha：学习率，iters：迭代次数
    输出参数
        theta：学习到的截距和斜率参数，j_history：迭代计算的J值历史
    """
    # 初始化
    n = len(y)  # 训练样本数
    j_history = np.zeros((iters * n,))

    it: int
    for it in range(iters):
        for j in range(n):
            # 用矩阵运算同时更新w参数
            xj = x[j].reshape(1, -1)
            yj = y[j].reshape(1, -1)
            theta = theta - alpha / n * (np.dot(xj.T, np.dot(xj, theta) - yj))
            # 保存代价J
            j_history[it * n + j] = compute_cost(x, y, theta, j)
    return theta, j_history


def plot_jhistory(j_history):
    """
    绘制代价J的下降曲线
    输入参数：
        j_history：J历史
    返回：
        无
    """
    plt.figure()
    plt.plot(j_history, 'r-')  # 绘制J历史曲线
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'代价J')

    plt.show()


def main():
    """ 主函数 """
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
    n: int = len(y_data)  # 样本数

    # 随机置乱
    np.random.seed(1234)
    idx = [i for i in range(n)]
    np.random.shuffle(idx)
    x_data = x_data[idx]
    y_data = y_data[idx]

    # 梯度下降
    # 添加一列全1，以扩展x
    x_data = np.column_stack((np.ones((n, 1)), x_data))
    theta = np.zeros((7, 1))  # 参数初始值

    # 梯度下降法的超参数设置
    iterations = 8  # 迭代次数
    alpha = 1  # 学习率

    # 调用梯度下降函数
    theta, j_history = stochastic_gradient_descent(x_data, y_data, theta, alpha, iterations)

    # 打印找到的参数
    print(f'梯度下降结果，Theta参数：\n {theta}')

    # 计算误差
    estimate_y = np.dot(x_data, theta)
    err = y_data - estimate_y
    rmse = np.squeeze(np.sqrt(np.dot(err.T, err) / n))
    print(f'\nRMSE：  {rmse} \n')

    # 绘图
    plot_jhistory(j_history)


# %%
if __name__ == "__main__":
    main()
