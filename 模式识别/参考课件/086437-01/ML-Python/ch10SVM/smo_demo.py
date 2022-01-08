# -*- coding: utf-8 -*-
"""
简化SMO示例
线性SVM使用Iris数据集
RBF SVM使用随机生成数据集

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import simple_smo
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


def generate_data(n):
    """
    生成数据集
    输入
        n：样本数
    输出
        data：n行3列数据集。前两列为数据，第三列为标签
    """
    np.random.seed(1)
    data = 10 * np.random.rand(n, 3) - 5
    for i in range(n):
        if np.square(data[i, 0]) / np.square(3.5) + np.square(data[i, 1]) / np.square(3) > 1:
            data[i, 2] = 1
        else:
            data[i, 2] = 0
        # 将边界上的数据随机加些噪声
        if np.square(np.square(data[i, 0]) / np.square(3.5) + np.square(data[i, 1]) / np.square(3) - 1) < 0.08:
            if np.random.rand(1) > 0.5:
                data[i, 2] = 1
            else:
                data[i, 2] = 0
    return data


def visualize_iris_boundary(x, y, model, c):
    """
    可视化鸢尾花二维散点图
    正例可视化为+，负例可视化为o
    输入
        x：特征，假设为N行2列矩阵
        y：类别标签，假设为1或0
        model：训练好的SVM模型
        c：SVM参数
    """
    plt.figure()
    neg_x = x[np.where(y[:, 0] == 0)]
    pos_x = x[np.where(y[:, 0] == 1)]
    neg = plt.scatter(neg_x[:, 0], neg_x[:, 1], c='r', marker='o')
    pos = plt.scatter(pos_x[:, 0], pos_x[:, 1], c='g', marker='+')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    # 只需要两点便可以定义一条直线，选择两个端点
    xplot = np.linspace(min(x[:, 0]), max(x[:, 0]))
    # 计算决策边界线，b + w1*x + w2*y = 0
    # 已知x，可计算y
    yplot = - (model['w'][0] * xplot + model['b']) / model['w'][1]
    # 绘制决策边界线
    boundary, = plt.plot(xplot, yplot, '-b')
    plt.ylim([-1, 3])
    plt.legend([neg, pos, boundary], ['Setosa', 'Versicolor', u'决策边界'], loc='lower right')
    plt.title(f'C = {c}')
    plt.show()


def visualize_boundary(x, y, model, c, sigma):
    """
    绘制SVM学习到的非线性决策边界
    正例可视化为+，负例可视化为o
    输入
        x：特征，假设为N行2列矩阵
        y：类别标签，假设为1或0
        model：训练好的SVM模型
        c、sigma：SVM参数
    """
    plt.figure()
    neg_x = x[np.where(y[:, 0] == 0)]
    pos_x = x[np.where(y[:, 0] == 1)]
    plt.scatter(neg_x[:, 0], neg_x[:, 1], c='r', marker='o', label="y=0")
    plt.scatter(pos_x[:, 0], pos_x[:, 1], c='g', marker='+', label="y=1")
    plt.xlabel('x1')
    plt.ylabel('x2')

    # 绘制决策边界
    # 网格范围
    u = np.linspace(min(x[:, 0]), max(x[:, 0]), 150)
    v = np.linspace(min(x[:, 1]), max(x[:, 1]), 150)
    uu, vv = np.meshgrid(u, v)  # 生成网格数据
    z = simple_smo.smo_predict(model, np.column_stack((uu.ravel(), vv.ravel())))
    # 将标签0转换为-1
    z[np.where(z[:, 0] == 0), 0] = -1
    # 保持维度一致
    z = z.reshape(uu.shape)
    # 画图
    plt.contour(uu, vv, z, 0)

    plt.legend(loc='lower right')
    plt.title(f'C = {c}, sigma={sigma}')
    plt.show()


def main():
    # 1、使用线性SVM
    # 加载数据
    file_path = "../data/fisheriris.csv"
    x, _ = read_csv(file_path)
    x = x[:100]

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    # 3、4列分别是petal length和petal width
    # 仅使用这两个属性
    x_data = x_data[:, 2:4]

    # 目标setosa为0，versicolor为1
    y_data = np.row_stack((np.zeros((50, 1)), np.ones((50, 1))))

    # 训练线性SVM
    # 可尝试改变C值，探索该值对决策边界的影响
    c = 100
    model = simple_smo.smo_train(x_data, y_data, c, simple_smo.linear_kernel, 1e-3, 20)
    # 可视化
    visualize_iris_boundary(x_data, y_data, model, c)

    # 计算训练准确率
    y_hat = simple_smo.smo_predict(model, x_data)
    print('\n训练准确率： {:.2%}\n'.format(np.mean(y_hat == y_data)))

    # 2、使用RBF SVM
    # 随机生成数据
    data = generate_data(250)
    x = data[:, [0, 1]]  # 前两列为属性x
    y = data[:, 2].reshape(-1, 1)  # 第3列为目标属性y

    # 初始化SVM参数
    # 尝试修改C和sigma参数
    c = 1
    sigma = 0.1

    # 数值公差和最大迭代次数都为默认值
    model = simple_smo.smo_train(x, y, c, simple_smo.gaussian_kernel)
    visualize_boundary(x, y, model, c, sigma)

    # 计算训练准确率
    y_hat = simple_smo.smo_predict(model, x)
    print('\n训练准确率： {:.2%}\n'.format(np.mean(y_hat == y)))
    print('\n程序结束。\n')


if __name__ == "__main__":
    main()
