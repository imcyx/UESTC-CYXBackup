# -*- coding: utf-8 -*-
"""
SMO算法的简化版本

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def linear_kernel(x1, x2):
    """
    返回x1和x2的线性核
    输入
        x1、x2：输入向量
    输出
        sim：两个向量的相似度
    """
    # 确保x1和x2都是列向量
    xx1 = x1.reshape(-1, 1)
    xx2 = x2.reshape(-1, 1)

    # 计算点积
    sim = np.dot(xx1.T, xx2)
    return sim


def gaussian_kernel(x1, x2, sigma=0.1):
    """
    返回x1和x2的高斯核
    输入
        x1、x2：输入向量，sigma：方差
    输出
        sim：两个向量的相似度
    """
    # 确保x1和x2都是列向量
    xx1 = x1.reshape(-1, 1)
    xx2 = x2.reshape(-1, 1)

    sim = np.exp(np.dot(-(xx1 - xx2).T, (xx1 - xx2)) / (2 * np.square(sigma)))
    return sim


def smo_train(x_data, y_data, c, kernel_function, tol=1e-3, max_passes=5):
    """
    使用SMO算法的简化版本训练SVM分类器
    输入
        x_data：特征矩阵。第i行为第i个训练样本，第j列为第j个特征
        y_data：目标属性。1为正例，0为负例
        c：标准SVM的正则化参数
        kernel_function：核函数。选项有：'linearKernel'、'gaussianKernel'或其他函数（预计算的核矩阵）
        tol：浮点数的数值公差
        max_passes：遍历alpha其值不改变的最大迭代次数
    输出
        model：训练好的模型
    """
    x = x_data
    y = y_data.copy()

    np.random.seed(1234)
    # 样本数
    n = len(x)

    # 把0改为-1，SVM算法要求
    y[np.where(y[:, 0] == 0), 0] = -1

    # 初始化各个变量
    alphas = np.zeros((n, 1))
    b = 0
    e = np.zeros((n, 1))
    passes = 0
    eta = 0
    l = 0
    h = 0

    # 使用向量化的核，加速SVM训练
    if kernel_function.__name__ == 'linear_kernel':
        # 线性核
        # 等价于每对样本计算一次核
        k = np.dot(x, x.T)
    elif kernel_function.__name__ == 'gaussian_kernel':
        # 向量化的RBF核
        x2 = np.sum(np.square(x), axis=1, keepdims=True)
        k = x2 + (x2.T - 2 * np.dot(x, x.T))
        k = np.power(kernel_function(np.array(1), np.array(0)), k)
    else:
        # 预计算的核矩阵。没有向量化，速度慢
        k = np.zeros(n)
        for i in range(n):
            for j in range(n):
                k[i, j] = kernel_function(x[i].T, x[j].T)
                k[j, i] = k[i, j]  # 对称矩阵

    # 训练
    dots = 0  # 提示正在迭代计算的点符号

    # CS229课程给出的简化SMO算法
    while passes < max_passes:
        alphas_changed = 0
        for i in range(n):
            # 步骤1，选择优化对象alpha i和alpha j
            # 计算误差 Ei = f(x(i)) - y(i)
            e[i, 0] = np.sum(alphas * y * k[:, i].reshape(-1, 1)) + b - y[i, 0]

            if (y[i, 0] * e[i, 0] < - tol and alphas[i] < c) or (y[i, 0] * e[i, 0] > tol and alphas[i, 0] > 0):
                # 实践中有很多选择i和j的启发式方法，这里简化为仅随机选择不等于i的j
                j = int(np.ceil(n * np.random.rand()))
                # 确保j不等于i
                while j == i or j >= n:
                    j = int(np.ceil(n * np.random.rand()))

                # 计算 Ej = f(x(j)) - y(j)
                e[j, 0] = np.sum(alphas * y * k[:, j].reshape(-1, 1)) + b - y[j, 0]

                # 保存旧的alpha
                old_alpha_i = alphas[i, 0]
                old_alpha_j = alphas[j, 0]

                # 计算 L 和 H
                if y[i, 0] == y[j, 0]:
                    l = np.max((0, alphas[j, 0] + alphas[i, 0] - c))
                    h = np.min((c, alphas[j, 0] + alphas[i, 0]))
                else:
                    l = np.max((0, alphas[j, 0] - alphas[i, 0]))
                    h = np.min((c, c + alphas[j, 0] - alphas[i, 0]))

                if l == h:
                    # continue到下一个i
                    continue

                # 步骤2，优化alpha i和alpha j
                # 计算eta
                eta = 2 * k[i, j] - k[i, i] - k[j, j]
                if eta >= 0:
                    # continue到下一个i
                    continue

                # 计算alpha j
                alphas[j, 0] = alphas[j, 0] - (y[j, 0] * (e[i, 0] - e[j, 0])) / eta

                # 将alpha j的值Clip到L至H范围
                alphas[j, 0] = np.min((h, alphas[j, 0]))
                alphas[j, 0] = np.max((l, alphas[j, 0]))

                # 检查alpha的变化是否显著
                if np.abs(alphas[j, 0] - old_alpha_j) < tol:
                    # 变化不显著则替换，并continue到下一个i
                    alphas[j, 0] = old_alpha_j
                    continue

                # 计算 alpha i
                alphas[i, 0] = alphas[i, 0] + y[i, 0] * y[j, 0] * (old_alpha_j - alphas[j, 0])

                # 步骤3，计算截距b
                # 计算 b1 b2
                b1 = b - e[i, 0] \
                     - y[i, 0] * (alphas[i, 0] - old_alpha_i) * k[i, j].T \
                     - y[j, 0] * (alphas[j, 0] - old_alpha_j) * k[i, j].T
                b2 = b - e[j, 0] \
                     - y[i, 0] * (alphas[i, 0] - old_alpha_i) * k[i, j].T \
                     - y[j, 0] * (alphas[j, 0] - old_alpha_j) * k[j, j].T

                # 计算 b
                if 0 < alphas[i, 0] < c:
                    b = b1
                elif 0 < alphas[j, 0] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                alphas_changed += 1

        if alphas_changed == 0:
            passes += 1
        else:
            passes = 0

        print('.', end='')
        dots += 1
        # 每迭代50趟换行
        if dots > 50:
            dots = 0
            print()

    # 保存训练好的模型
    idx = np.where(alphas > 0)[0]
    model = {}
    model['x'] = x[idx]
    model['y'] = y[idx]
    model['kernel_function'] = kernel_function
    model['b'] = b
    model['alphas'] = alphas[idx]
    model['w'] = (np.dot((alphas * y).T, x)).T

    return model


def smo_predict(model, x):
    """
    使用训练好的SVM分类器模型进行预测
    输入
        model：训练好的模型
        x：预测数据集。第i列行为第i个训练样本，第j列为第j个特征
    输出
        y：预测结果，N x 1，取值{0, 1}
    """
    # 初始化
    n = len(x)
    p = np.zeros((n, 1))
    y = np.zeros((n, 1))

    if model['kernel_function'].__name__ == 'linear_kernel':
        # 线性核，直接使用权重和截距
        p = np.dot(x, model['w']) + model['b']
    elif model['kernel_function'].__name__ == 'gaussian_kernel':
        # 向量化的RBF核。等价于在每对样本计算核
        x1 = np.sum(np.square(x), 1, keepdims=True)
        x2 = np.sum(np.square(model['x']), 1, keepdims=True).T
        k = x1 + (x2 - 2 * np.dot(x, model['x'].T))
        k = np.power(model['kernel_function'](np.array(1), np.array(0)), k)
        k = model['y'].T * k
        k = model['alphas'].T * k
        p = np.sum(k, 1, keepdims=True)
    else:
        # 其他非线性核
        for i in range(n):
            prediction = 0
            for j in range(len(model['x'])):
                prediction += model['alphas'][j] * model['y'][j] * \
                              model['kernel_function'](x[i, :].T, model.x[j, :].T)
                p[i] = prediction + model['b']

    # 将预测结果转换为{0, 1}范围
    y[np.where(p[:, 0] >= 0), 0] = 1
    y[np.where(p[:, 0] < 0), 0] = 0

    return y
