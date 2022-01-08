# -*- coding: utf-8 -*-
"""
绘制学习曲线

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def generate_data():
    """
    生成数据
    """
    np.random.seed(1234)
    n = 200  # 样本数
    # 生成0到1之间的x
    x = np.random.rand(n, 1)
    noise = 0.2          # 噪声强度
    # 目标
    y = np.sin(2 * np.pi * x) + noise * np.random.randn(len(x), 1)
    # 验证集
    training_size = int(round(n / 2))
    val_x = x[training_size:]
    val_y = y[training_size:]
    # 训练集
    train_x = x[:training_size]
    train_y = y[:training_size]
    return train_x, train_y, val_x, val_y


def poly_features(x, degree):
    """
    将两个特征转换为多项式特征
    例如，将x转换为1, x, x.^2, x.^3, ...
    输入
        x：n行1列特征，degree：阶次
    输出
        x_data：转换后的多项式特征
    """
    x_data = np.ones((len(x), 1))  # 截距项
    for i in range(1, degree + 1):
        x_data = np.column_stack((x_data, np.power(x, i)))

    return x_data


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


def plot_learning_curves(train_loss, ind_loss, ns, order):
    """
    绘制训练损失和验证损失曲线
    """
    plt.figure()
    plt.plot(ns, train_loss, 'b--', label=u'训练误差')
    plt.plot(ns, ind_loss, 'g-', label=u'验证误差')
    plt.xlabel(u'N（训练集大小）')
    plt.ylabel(u'误差')
    plt.ylim([0, 2])
    plt.xlim([0, 100])
    plt.title(f'学习曲线（{order}次模型）')
    plt.legend()
    plt.show()


def model_train(train_x, train_y, val_x, val_y):
    """
    模型训练
    """
    max_order = 3  # 最高模型阶次

    # 训练集大小
    ns = np.arange(1, len(train_y))
    # 独立验证集损失
    val_loss = np.zeros(len(ns))
    # 训练损失
    train_loss = np.zeros(len(ns))

    # 按模型阶次顺序运行训练和验证
    for order in range(1, max_order + 1):
        # 构建训练集和测试集矩阵
        poly_train_x = poly_features(train_x, order + 1)
        poly_val_x = poly_features(val_x, order + 1)

        for idx in range(len(ns)):
            # train_x_lim为指定size的训练集
            train_x_lim = poly_train_x[:ns[idx]]
            train_y_lim = train_y[:ns[idx]]

            # 正规方程求回归参数
            theta = normal_equation(train_x_lim, train_y_lim)
            # 预测
            train_pred_lim = np.dot(train_x_lim, theta)
            train_loss[idx] = np.mean(np.square(train_pred_lim - train_y_lim))
            val_pred = np.dot(poly_val_x, theta)
            val_loss[idx] = np.mean(np.square(val_pred - val_y))

        # 绘制学习曲线
        plot_learning_curves(train_loss, val_loss, ns, order)
    return None


def main():
    np.random.seed(2)
    # 加载数据
    train_x, train_y, val_x, val_y = generate_data()
    # 训练并绘制学习曲线
    model_train(train_x, train_y, val_x, val_y)


if __name__ == "__main__":
    main()
