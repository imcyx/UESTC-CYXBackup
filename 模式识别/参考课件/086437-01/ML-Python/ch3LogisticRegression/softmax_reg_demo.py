# -*- coding: utf-8 -*-
"""
Softmax回归示例
使用鸢尾花数据集
todo

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
        x_data：属性，y：目标属性
    """
    with open(file, encoding="utf-8") as fr:
        fr.readline()  # 跳过标题行
        content = fr.readlines()
        x_data = [f.split(",")[: -1] for f in content]
        y = [f.split(",")[-1].strip("\n") for f in content]
    return x_data, y


def softmax(z):
    """
    Numpy实现Softmax激活函数。每一个Z减去一个max值是为了避免数值溢出
    输入参数：
        z：二维Numpy数组
    返回：
        a：softmax(z)输出，与z的shape一致
    """
    z_rescale = z - np.max(z, axis=1, keepdims=True)
    a = np.exp(z_rescale) / np.sum(np.exp(z_rescale), axis=1, keepdims=True)
    assert (a.shape == z.shape)
    return a


def cost_function(x, y, theta, my_lambda, k):
    """
    计算正则化Softmax回归的代价和梯度
    输入
        x：特征矩阵，y：目标属性，theta：参数集，lambda：正则化参数， k：类别数
    输出
        j_value：代价，grad：梯度
    """
    n, d1 = len(x), len(x[0])
    temp_theta = theta.copy()
    temp_theta[:, 0] = 0      # 正则化不包括截距项
    one_hot_y = np.eye(k)[y.reshape(-1).astype(int)]       # 将y变换为(n, k)矩阵，每行为标签的one-hot编码
    z = softmax(np.dot(x, theta.T))        # x(n, d1) * theta.T(d1, k) = p(n, k)
    j_value = -1 / n * np.sum(np.multiply(one_hot_y, np.log(z))) + my_lambda / 2 * np.sum(np.power(temp_theta, 2))
    grad = -1 / n * np.dot((one_hot_y - z).T, x) + my_lambda * temp_theta
    return j_value, grad


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


def predict(theta, new_x):
    """
    使用学习到的softmax回归参数theta来预测新样本new_x的标签
    输入参数：
        theta：softmax回归参数，new_x：新样本集
    返回：
        y_hat：预测的k种类别
    """
    pred = np.dot(new_x, theta.T)
    y_hat = np.argmax(pred, axis=1)
    return y_hat.reshape(-1, 1)


def softmax_train(x, y, alpha, iters, my_lambda, k):
    """
    训练
    输入
        x：特征矩阵，y：目标属性，alpha：学习率，iters：迭代次数，my_lambda：正则化参数， k：类别数
    输出
        theta：优化的参数集
    """
    n = len(x)
    d1 = len(x[0])  # d plus 1
    # 初始化参数
    theta = 0.005 * np.random.randn(k, d1)
    j_history = np.zeros((iters,))
    for it in range(iters):
        j_history[it], grad = cost_function(x, y, theta, my_lambda, k)
        # 更新theta参数
        theta = theta - alpha / n * grad

    return theta, j_history


def main():
    # 梯度下降法的设置
    alpha = 5  # 学习率
    iterations = 800  # 迭代次数
    my_lambda = 0.05    # 正则化参数
    k = 3       # 类别数

    #  加载鸢尾花数据
    file_path = "../data/fisheriris.csv"
    x, _ = read_csv(file_path)

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    # 目标setosa为0，versicolor为1，virginica为2
    y_data = np.row_stack((np.zeros((50, 1)), np.ones((50, 1)), np.ones((50, 1)) * 2))
    n = len(y_data)  # 样本数

    # 加截距项
    x_data = np.column_stack((np.ones((n, 1)), x_data))

    theta, j_history = softmax_train(x_data, y_data, alpha, iterations, my_lambda, k)
    print(theta)

    plot_jhistory(j_history)

    # 计算在训练集上的分类正确率
    y_hat = predict(theta, x_data)
    print('训练集上的分类正确率：{:.2%}'.format(np.mean(y_hat == y_data)))


if __name__ == "__main__":
    main()
