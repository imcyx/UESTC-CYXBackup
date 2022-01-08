# -*- coding: utf-8 -*-
"""
逻辑异或示例2
使用神经网络来模拟XOR逻辑

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


def get_data():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float)  # 两列为属性x1、x2
    y = np.array([0, 1, 1, 0], dtype=np.float)
    return x, y


def sigmoid(z):
    """ S型激活函数 """
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoid_gradient(z):
    """ 计算Sigmoid激活函数的导数 """
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def predict(w1, w2, new_x):
    """
    给定训练好的网络参数w1和w2，预测测试集new_x的标签
    """
    n = len(new_x)

    h1 = sigmoid(np.dot(np.column_stack((np.ones((n, 1)), new_x)), w1.T))
    h2 = sigmoid(np.dot(np.column_stack((np.ones((n, 1)), h1)), w2.T))
    p = h2.ravel()
    return p


def rand_init_weights(fan_in, fan_out, my_epsilon=0.1):
    """
    随机初始化神经网络中一层的权重
    输入
        fan_in：输入连接数
        fan_out：输出连接数
        epsilon_init：从均匀分布中得到的权重的取值范围
    输出
        init_w：权重初始化为随机值。 应为(fan_out, 1 + fan_in)的矩阵，第一列为偏置项
    """
    init_w = np.random.rand(fan_out, 1 + fan_in) * 2 * my_epsilon - my_epsilon
    return init_w


def cost_function(params, num_input_units, num_hidden_units, num_labels,
                  x, y, my_lambda=0.0):
    """
    实现两层神经网络代价函数和梯度的计算
    输入
        params：参数向量，这是优化函数的要求，计算代价时需要转换为权重矩阵theta1和theta2
        num_input_units：输入单元数
        num_hidden_units：第二层的隐藏单元数
        num_labels：标签数
        x：数据集特征
        y：标签
        my_lambda：正则化参数
    输出
        j：代价函数值
        grad：梯度，theta1和theta2的偏导数
    """
    # 从网络参数中获取w1和w2
    w1 = np.reshape(params[:num_hidden_units * (num_input_units + 1)],
                    (num_hidden_units, (num_input_units + 1)))
    w2 = np.reshape(params[(num_hidden_units * (num_input_units + 1)):],
                    (num_labels, (num_hidden_units + 1)))

    # 样本数
    n = len(y)

    # 没有必要将向量y转换为独热码
    one_hot_y = y.reshape((-1, 1))
    # 第1步，前向传播
    eps = 1e-6      # 避免数值计算问题
    a1 = np.column_stack((np.ones((n, 1)), x))
    z2 = np.dot(a1, w1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((len(a2), 1)), a2))
    z3 = np.dot(a2, w2.T)
    h = sigmoid(z3)
    j = 1 / n * np.sum(np.multiply(-one_hot_y, np.log(h + eps)) - np.multiply((1 - one_hot_y), np.log(1 - h + eps)))

    # 第2步，反向传播
    delta_3 = h - one_hot_y
    delta_2 = np.dot(delta_3, w2[:, 1:]) * sigmoid_gradient(z2)
    w2_grad = 1 / n * np.dot(delta_3.T, a2)
    w1_grad = 1 / n * np.dot(delta_2.T, a1)

    # 第2步，正则化
    temp_w1 = w1.copy()
    temp_w2 = w2.copy()
    temp_w1[:, 0] = 0
    temp_w2[:, 0] = 0
    j = j + my_lambda / 2 * (np.sum(np.square(temp_w1)) + np.sum(np.square(temp_w2)))
    w1_grad += my_lambda * temp_w1
    w2_grad += my_lambda * temp_w2

    grad = np.concatenate([w1_grad.ravel(), w2_grad.ravel()])

    return j, grad


def plot_decision_boundary(w1, w2, x, y):
    """
    绘制散点图
    输入参数：
        w1、w2：网络回归参数，x：特征，y：目标属性
    返回：
        无
    """
    plt.figure()
    neg_x = x[np.where(y == 0)]
    pos_x = x[np.where(y == 1)]
    neg = plt.scatter(neg_x[:, 0], neg_x[:, 1], c='r', marker='o')
    pos = plt.scatter(pos_x[:, 0], pos_x[:, 1], c='b', marker='+')

    # 绘制决策边界
    # 网格范围
    u = np.linspace(min(x[:, 0]), max(x[:, 0]), 150)
    v = np.linspace(min(x[:, 1]), max(x[:, 1]), 150)
    uu, vv = np.meshgrid(u, v)  # 生成网格数据
    z = predict(w1, w2, np.column_stack((uu.ravel(), vv.ravel())))
    # 保持维度一致
    z = z.reshape(uu.shape)
    # 画图
    plt.contour(uu, vv, z, 0)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend([neg, pos], ['y=0', 'y=1'])
    plt.title(u'逻辑异或')

    plt.show()


def main():
    np.random.seed(1)

    # 训练数据
    x, y = get_data()

    # 网络结构
    input_units = len(x[0])
    hidden_units = 2   # 隐藏单元数
    num_labels = 1         # 标签数

    init_w1 = rand_init_weights(input_units, hidden_units)
    init_w2 = rand_init_weights(hidden_units, num_labels)

    # 合二为一
    init_params = np.concatenate([init_w1.ravel(), init_w2.ravel()], axis=0)

    # 可以修改下面的正则化参数
    my_lambda = 0.0001

    # 方便调用优化函数
    def c_f(p):
        return cost_function(p, input_units, hidden_units,
                             num_labels, x, y, my_lambda)

    # 优化迭代次数
    options = {'maxiter': 200}
    res = optimize.minimize(c_f, init_params, jac=True, method='TNC', options=options)

    opt_params = res.x

    # 分解网络参数
    w1 = np.reshape(opt_params[:hidden_units * (input_units + 1)],
                        (hidden_units, (input_units + 1)))
    w2 = np.reshape(opt_params[(hidden_units * (input_units + 1)):],
                        (num_labels, (hidden_units + 1)))

    # 计算在训练集上的分类正确率
    y_hat = predict(w1, w2, x) >= 0.5
    print(y_hat.astype(int))
    print('训练集上的分类正确率: {:.2%}'.format(np.mean(y_hat == y)))

    # 绘图
    plot_decision_boundary(w1, w2, x, y)


if __name__ == "__main__":
    main()
