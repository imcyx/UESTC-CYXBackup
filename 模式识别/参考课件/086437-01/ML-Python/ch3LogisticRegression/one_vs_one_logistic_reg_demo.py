# -*- coding: utf-8 -*-
"""
逻辑回归（Logistic Regression）分类
这里测试多元分类，使用一对一方式

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


def plot_data(x, y):
    """ 绘制鸢尾花二维数据集散点图 """
    plt.figure()
    setosa_x = x[np.where(y[:, 0] == 0)]
    versicolor_x = x[np.where(y[:, 0] == 1)]
    virginica_x = x[np.where(y[:, 0] == 2)]
    setosa = plt.scatter(setosa_x[:, 1], setosa_x[:, 2], c='r', marker='o')
    versicolor = plt.scatter(versicolor_x[:, 1], versicolor_x[:, 2], c='g', marker='x')
    virginica = plt.scatter(virginica_x[:, 1], virginica_x[:, 2], c='b', marker='+')

    # 坐标
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    # 图例
    plt.legend([setosa, versicolor, virginica], ['Setosa', 'Versicolor', 'Virginica'], loc='lower right')
    plt.show()


def poly_features(x1, x2):
    """
    将两个特征转换为多项式特征
    例如，将x1, x2转换为1, x1, x2, x1.^2, x2.^2, x1*x2, x1*x2.^2...
    输入
        x1，x2：两个特征都是N行1列
    输出
        x_data：转换后的多项式特征
    """
    degree = 5  # 阶次
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
    eps = 1e-6      # 避免数值计算问题
    j_value = -1 / n * np.add(np.dot(y.T, np.log(h + eps)), np.dot((1 - y).T, np.log(1 - h + eps))) + \
              my_lambda / 2 * np.dot(temp_theta.T, temp_theta)
    grad = 1 / n * np.dot(x.T, np.subtract(h, y)) + my_lambda * temp_theta
    return j_value, grad


def plot_decision_boundary(x, y, theta, j_value, pos_str, neg_str):
    """
    绘制决策边界
    输入参数
        x：输入，y：输出，theta：参数，j_value：代价，pos_str：正例显示字符串，neg_str：负例显示字符串
    输出参数
        无
    """
    neg_x = x[np.where(y[:, 0] == 0)]
    pos_x = x[np.where(y[:, 0] == 1)]
    plt.clf()
    neg = plt.scatter(neg_x[:, 1], neg_x[:, 2], c='b', marker='o')
    pos = plt.scatter(pos_x[:, 1], pos_x[:, 2], c='r', marker='+')

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

    # 坐标
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    # 图例
    plt.legend([neg, pos], [neg_str, pos_str], loc='lower right')
    plt.title(u'代价J=%f' % j_value)
    plt.pause(0.1)


def gradient_descent(x, y, theta, alpha, iters, my_lambda, pos_str, neg_str):
    """
    梯度下降函数，找到合适的参数
    输入参数
        x：输入，y：输出，theta：参数，alpha：学习率，iters：迭代次数，my_lambda：正则化参数
        pos_str：正例显示字符串，neg_str：负例显示字符串
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
            plot_decision_boundary(x, y, theta, j_history[it], pos_str, neg_str)
    # 不使用动画
    if not use_animation:
        plot_decision_boundary(x, y, theta, j_history[-1], pos_str, neg_str)
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


def predict(theta, new_x):
    """
    使用学习到的逻辑回归参数theta来预测新样本new_x的标签
    输入参数：
        theta：逻辑回归参数，new_x：新样本集
    返回：
        y_hat：预测的三种类别
    """
    n = len(new_x)
    pred = (sigmoid(np.dot(new_x, theta)) >= 0.5).astype(int)
    pred_not = 1 - pred
    vote = np.zeros((n, 3))
    # 统计投票
    vote[:, 0] = pred[:, 0] + pred[:, 1]
    vote[:, 1] = pred_not[:, 0] + pred[:, 2]
    vote[:, 2] = pred_not[:, 1] + pred_not[:, 2]
    y_hat = np.argmax(vote, axis=1)
    return y_hat.reshape(-1, 1)


def main():
    # 梯度下降法的设置
    alpha = 0.1  # 学习率
    iterations = 100000  # 迭代次数
    # 正则化参数lambda
    my_lambda = 0.01

    #  加载鸢尾花数据
    file_path = "../data/fisheriris.csv"
    x, _ = read_csv(file_path)

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    # 第3、4列分别是petal length和petal width，仅使用这两个属性
    x_data = x_data[:, [2, 3]]
    # 目标setosa为0，versicolor为1，virginica为2
    y_data = np.row_stack((np.zeros((50, 1)), np.ones((50, 1)), np.ones((50, 1)) * 2))
    n = len(y_data)  # 样本数

    # 添加一列全1，以扩展x
    x_data = np.column_stack((np.ones((n, 1)), x_data))

    # 绘制散点图
    plot_data(x_data, y_data)

    # 添加截距项和多项式项
    poly_x_data = poly_features(x_data[:, 1].reshape(-1, 1), x_data[:, 2].reshape(-1, 1))

    d = len(poly_x_data[0])

    # 3个二元分类器的theta
    theta = np.zeros((d, 3))
    # 代价
    j_values = []

    # 转换为二元分类问题，分别计算代价和梯度
    classifier_idx = 0  # 当前二元分类器索引
    for i in range(3):
        for j in range(i + 1, 3):
            pos_x = poly_x_data[np.where(y_data[:, 0] == i)]
            neg_x = poly_x_data[np.where(y_data[:, 0] == j)]
            temp_x = np.row_stack((pos_x, neg_x))
            temp_y = np.row_stack((np.ones((50, 1)), np.zeros((50, 1))))
            if classifier_idx == 0:
                pos_str, neg_str = 'Setosa', 'Versicolor'
            elif classifier_idx == 1:
                pos_str, neg_str = 'Setosa', 'Virginica'
            else:
                pos_str, neg_str = 'Versicolor', 'Virginica'

            # 梯度下降
            temp_theta = np.zeros((d, 1))
            temp_theta, j_history = \
                gradient_descent(temp_x, temp_y, temp_theta, alpha, iterations,
                                 my_lambda, pos_str, neg_str)

            # plot_jhistory(j_history)
            theta[:, classifier_idx] = temp_theta.ravel()
            j_values.append(j_history)
            classifier_idx += 1  # 下一个分类器

    # 计算在训练集上的分类正确率
    y_hat = predict(theta, poly_x_data)
    print('训练集上的分类正确率：{:.2%}'.format(np.mean(y_hat == y_data)))


if __name__ == "__main__":
    main()
