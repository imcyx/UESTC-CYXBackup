# -*- coding: utf-8 -*-
"""
使用神经网络解决手写字符识别问题
输出层使用Softmax激活函数

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.append('..')
from utils.mnist_read import load_mnist

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def sigmoid(z):
    """ Sigmoid激活函数 """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_gradient(z):
    """ 计算Sigmoid激活函数的导数 """
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


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


def display_mnist_data(x):
    """
    显示MNIST图像数据，默认图像高宽相等
    输入
        x：要显示的图像样本
    输出
        无
    """
    figsize = (8, 8)
    # 计算输入数据的行数和列数
    if x.ndim == 2:
        n, d = x.shape
    elif x.ndim == 1:
        d = x.size
        n = 1
        x = x.reshape(1, -1)
    else:
        raise IndexError('输入只能是一维或二维的图片样本集合。')

    img_width = int(np.round(np.sqrt(d)))
    img_height = int(d / img_width)

    # 计算要显示的图像的行数和列数
    display_rows = int(np.floor(np.sqrt(n)))
    display_cols = int(np.ceil(n / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    ax_array = [ax_array] if n == 1 else ax_array.ravel()

    # 循环显示图像
    for i, ax in enumerate(ax_array):
        ax.imshow(x[i].reshape(img_height, img_width, order='C'), cmap='gray')
        ax.axis('off')

    plt.show()


def predict(w1, w2, new_x):
    """
    给定训练好的网络参数w1和w2，预测测试集new_x的标签
    """
    n = len(new_x)

    h1 = sigmoid(np.dot(np.column_stack((np.ones((n, 1)), new_x)), w1.T))
    h2 = sigmoid(np.dot(np.column_stack((np.ones((n, 1)), h1)), w2.T))
    p = np.argmax(h2, axis=1)
    return p


def generate_debug_weights(fan_out, fan_in):
    """
    随机生成指定输出和输入连接数的权重参数
    注意权重应该包含偏置项
    输入
        fan_out：输出连接数
        fan_in：输入连接数
    输出
        init_w：形状为(1 + fan_in, fan_out)的随机矩阵
    """
    init_w = np.random.random((1 + fan_in) * fan_out) / 10.0
    init_w = init_w.reshape(fan_out, 1 + fan_in, order='F')
    return init_w


def compute_approx_gradient(cost_func, theta, eps=1e-4):
    """
    计算近似梯度
    输入
        cost_func:代价函数
        theta：给定的网络参数
        eps：epsilon
    输出
        approx_grad：近似梯度
    """
    approx_grad = np.zeros(theta.shape)
    delta = np.diag(eps * np.ones(theta.shape))
    for i in range(theta.size):
        loss_plus, _ = cost_func(theta + delta[:, i])
        loss_minus, _ = cost_func(theta - delta[:, i])
        approx_grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return approx_grad


def check_gradients(cost_func, my_lambda=0):
    """
    创建一个小型神经网络来检查反向传播梯度。 本函数会输出由反向传播代码和数值梯度计算产生的梯度，这两个不同的梯度计算应该产生非常近似的值。
    输入
        cost_func：实现的代价函数
        my_lambda：正则化参数
    输出
        无
    """
    input_units = 3
    hidden_units = 5
    num_labels = 3
    n = 6

    # 随机产生权重数据
    w1 = generate_debug_weights(hidden_units, input_units)
    w2 = generate_debug_weights(num_labels, hidden_units)

    # 训练数据也照此产生
    x = generate_debug_weights(n, input_units - 1)
    y = np.random.randint(num_labels, size=n)  # 随机生成标签
    # 参数合二为一
    params = np.concatenate([w1.ravel(), w2.ravel()])

    # 简短的代价函数
    def cost_f(p):
        return cost_func(p, input_units, hidden_units, num_labels, x, y, my_lambda)

    cost, grad = cost_f(params)
    approx_grad = compute_approx_gradient(cost_f, params)

    # 为了方便观察，将两种方法计算出来的梯度并列，两者值应该相近
    print('以下左列为近似梯度，右列为损失函数计算出来的梯度，两者应该接近。')
    print(np.column_stack((approx_grad, grad)))

    diff = np.linalg.norm(approx_grad - grad) / np.linalg.norm(approx_grad + grad)
    print(f'计算两种实现的差，如果正确实现，差值应该小于1e-9\n差值：{diff}')


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
        params：参数向量，这是优化函数的要求，计算代价时需要转换为权重矩阵w1和w2
        num_input_units：输入单元数
        num_hidden_units：第二层的隐藏单元数
        num_labels：标签数
        x：数据集特征
        y：标签
        my_lambda：正则化参数
    输出
        j：代价函数值
        grad：梯度，w1和w2的偏导数
    """
    # 从网络参数中获取w1和w2
    w1 = np.reshape(params[:num_hidden_units * (num_input_units + 1)],
                    (num_hidden_units, (num_input_units + 1)))
    w2 = np.reshape(params[(num_hidden_units * (num_input_units + 1)):],
                    (num_labels, (num_hidden_units + 1)))

    # 样本数
    n = len(y)

    # 将向量y转换为独热码
    one_hot_y = np.eye(num_labels)[y.astype(int)]
    # 第1步，前向传播
    a1 = np.column_stack((np.ones((n, 1)), x))
    z2 = np.dot(a1, w1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((len(a2), 1)), a2))
    z3 = np.dot(a2, w2.T)
    h = softmax(z3)
    j = -1 / n * np.sum(np.multiply(one_hot_y, np.log(h)))

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


def main():
    #  加载MNIST数据
    x, y = load_mnist('../data/mnist/')
    test_x, test_y = load_mnist('../data/mnist/', train=False)

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]
    test_x_data = np.zeros((len(test_x), len(test_x[0])))
    for i in range(len(test_x_data[0])):
        test_x_data[:, i] = [float(f[i]) for f in test_x]

    # 样本数
    n = len(y)

    # 随机选择100个样本来显示
    rand_idx = np.random.choice(n, 100, replace=False)
    disp_samples = x_data[rand_idx, :]

    display_mnist_data(disp_samples)

    # 规范化
    x_data /= 255
    test_x_data /= 255

    # 网络结构
    input_units = len(x_data[0])
    hidden_units = 30  # 隐藏单元数
    num_labels = 10         # 标签数

    init_w1 = rand_init_weights(input_units, hidden_units)
    init_w2 = rand_init_weights(hidden_units, num_labels)

    # 将两部分参数合二为一
    init_params = np.concatenate([init_w1.ravel(), init_w2.ravel()], axis=0)

    # 可以修改下面的正则化参数
    my_lambda = 8e-6

    # 梯度检验
    # 检验代价函数正确编码以后，可以注释以下两句
    check_gradients(cost_function, my_lambda)

    # 方便调用优化函数
    def c_f(p):
        return cost_function(p, input_units, hidden_units,
                             num_labels, x_data, y, my_lambda)

    # 优化迭代次数
    options = {'maxiter': 400}
    res = optimize.minimize(c_f, init_params, jac=True, method='TNC', options=options)

    opt_params = res.x

    # 分解网络参数
    w1 = np.reshape(opt_params[:hidden_units * (input_units + 1)],
                        (hidden_units, (input_units + 1)))
    w2 = np.reshape(opt_params[(hidden_units * (input_units + 1)):],
                        (num_labels, (hidden_units + 1)))

    # 计算在训练集上的分类正确率
    y_hat = predict(w1, w2, x_data)
    print('训练集上的分类正确率: {:.2%}'.format(np.mean(y_hat == y)))

    # 计算在测试集上的分类正确率
    test_y_hat = predict(w1, w2, test_x_data)
    print('测试集上的分类正确率: {:.2%}'.format(np.mean(test_y_hat == test_y)))


if __name__ == "__main__":
    main()
