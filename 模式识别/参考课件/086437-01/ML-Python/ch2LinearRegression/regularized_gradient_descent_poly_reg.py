# -*- coding: utf-8 -*-
"""
正则化多项式回归拟合历届奥运会自由泳100米数据
使用梯度下降法

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
    # 排除全1的第一列
    mu[0][0] = 0
    sigma[0][0] = 1
    normalized_x = (x - mu) / sigma
    return normalized_x, mu, sigma


def compute_cost(x, y, theta, my_lambda):
    """
    计算线性回归的代价
    使用w作为线性回归的参数，计算代价J
    输入参数
        x：输入，y：输出，theta：参数，my_lambda：正则化参数
    输出参数
        j_value：计算的J值，grad:梯度
    """
    n = len(y)  # 训练样本数
    # 不规范化theta0项
    temp_theta = theta.copy()
    temp_theta[0, 0] = 0

    j_value = 1 / (2 * n) * np.sum(np.power(np.subtract(np.dot(x, theta), y), 2)) + \
              my_lambda / 2 * np.dot(temp_theta.T, temp_theta)
    grad = 1 / n * np.dot(x.T, (np.dot(x, theta) - y))
    grad = grad + np.dot(my_lambda, temp_theta)
    return j_value, grad


def gradient_descent(x, y, theta, alpha, iters, my_lambda):
    """
    梯度下降函数，找到合适的参数
    输入参数
        x：输入，y：输出，theta：参数，alpha：学习率，iters：迭代次数，my_lambda：正则化参数
    输出参数
        theta：学习到的参数，j_history：迭代计算的J值历史
    """
    # 初始化
    j_history = np.zeros((iters,))

    for it in range(iters):
        j_value, grad = compute_cost(x, y, theta, my_lambda)
        # 用矩阵运算同时更新theta参数
        theta = theta - alpha * grad

        # 保存代价J
        j_history[it] = j_value
    return theta, j_history


def plot_scatter(x, y, theta, mu, sigma):
    """
    绘制散点图
    输入参数：
        x：一个属性，y：目标属性，theta：参数
    返回：
        无
    """
    plt.figure()
    plt.scatter(x, y, c='b', marker='o', s=5, linewidths=2)
    lim = np.array([min(x) - 4, max(x) + 4])  # 为上下限留一点空隙
    plt.xlim(lim)  # 设定x坐标轴的范围
    # 绘制拟合线
    rank: int = len(theta) - 1
    # 显示的数据点。左右留边距2
    p_x = np.arange(min(x) - 2, max(x) + 2, 0.01).reshape(-1, 1)
    plot_x = np.ones((len(p_x), 1))
    for i in range(rank):
        plot_x = np.column_stack((plot_x, np.power(p_x, i+1)))
    # 对plot_x也同样正则化
    plot_x = (plot_x - mu) / sigma
    plt.plot(p_x, np.dot(plot_x, theta).reshape(-1, 1), 'r')
    plt.xlabel(u'奥运会举办年编号')
    plt.ylabel(u'取胜时间（秒）')
    plt.title(u'正则化%d次模型' % rank)

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
    # 梯度下降法的超参数设置
    iterations = 100  # 迭代次数
    alpha = 0.2  # 学习率
    my_lambda = 0.02  # 正则化参数
    order = 8  # 八次模型

    # 加载奥运会数据
    file_path = "../data/Freestyle100m.csv"
    x, y = read_csv(file_path)

    x = [float(f[0]) for f in x]
    # 转换为Numpy数组
    x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
    # 将奥运会举办年转换为届，避免数值计算问题
    x -= x[0]  # 举办年都减去最初的年份1896
    x /= 4  # 4年一届

    n = len(y)  # 样本数

    x_data = np.ones((n, 1))
    for i in range(order):
        x_data = np.column_stack((x_data, np.power(x, i + 1)))

    # 特征规范化
    x_data, mu, sigma = feature_normalize(x_data)

    theta = np.zeros((order + 1, 1))  # 参数初始值

    # 调用梯度下降函数
    theta, j_history = gradient_descent(x_data, y, theta, alpha, iterations, my_lambda)

    # 打印找到的参数
    print(f'梯度下降结果，Theta参数：\n {theta}')

    # 计算误差
    estimate_y = np.dot(x_data, theta)
    err = y - estimate_y
    rmse = np.squeeze(np.sqrt(np.dot(err.T, err) / n))
    print(f'\nRMSE：  {rmse} \n')

    # 绘图
    plot_scatter(x, y, theta, mu, sigma)
    plot_jhistory(j_history)


if __name__ == "__main__":
    main()
