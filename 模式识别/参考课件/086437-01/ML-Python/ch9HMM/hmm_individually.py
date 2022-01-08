# -*- coding: utf-8 -*-
"""
HMM求取单个最有可能状态算法实现
计算马尔科夫链最有可能的状态序列

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def hmm_individually(a, b, pi, o):
    """
    输入：
        a：转移概率矩阵
        b：发射概率矩阵
        pi：初始状态分布
        o：观测的输出序列
    输出：
        path：最佳状态路径
        gamma：概率矩阵历史。t行是t时刻的gamma
    """
    n = len(a)
    t = len(o)
    alpha = np.zeros((t, n))
    beta = np.zeros((t, n))
    gamma = np.zeros((t, n))
    path = np.zeros(t)

    # 1、初始化
    alpha[0, :] = pi * b[:, o[0]]
    beta[t - 1, :] = 1

    # 2、计算alpha和beta
    for k in range(t - 1):
        for j in range(n):
            alpha[k + 1, j] = np.dot(alpha[k, :], a[:, j]) * b[j, o[k + 1]]

    for k in range(t - 2, -1, -1):
        for j in range(n):
            for i in range(n):
                beta[k, i] += a[i, j] * b[j, o[k + 1]] * beta[k + 1, j]

    # 3、迭代求gamma
    for k in range(t):
        norm = np.dot(alpha[k, :], beta[k, :])
        gamma[k, :] = alpha[k, :] * beta[k, :] / norm

    # 4、终结
    for k in range(t):
        path[k] = np.argmax(gamma[k, :])

    return path, gamma
