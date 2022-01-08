# -*- coding: utf-8 -*-
"""
HMM学习算法实现
给定一定观测序列，识别马尔科夫链的参数

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def hmm_learn(o, a_init, b_init, pi_init):
    """
    输入：
        o：观测的输出序列，每一行为HMM的一次输出序列
        a_init：初始状态转移矩阵
        b_init：初始发射矩阵
        pi_init：初始状态分布
    输出：
        a：估计的转移概率矩阵
        b：估计的发射概率矩阵
        pi：估计的初始状态分布
    """
    a = a_init.copy()
    a_old = a.copy()
    b = b_init.copy()
    b_old = b.copy()
    pi = pi_init.copy()
    pi_old = pi.copy()
    n = len(a)
    m = len(b[0])
    l, t = len(o), len(o[0])
    alpha = np.zeros((t, n, l))
    beta = np.zeros((t, n, l))
    gamma = np.zeros((t, n, l))
    xi = np.zeros((t - 1, n, n, l))
    iters = 1000  # 迭代次数

    for it in range(iters):
        # 1、初始化
        for li in range(l):
            alpha[0, :, li] = pi_old * b_old[:, o[li, 0]]
            beta[t - 1, :, li] = 1

        # 2、计算alpha、beta、gamma和xi
        for li in range(l):
            # 计算alpha
            for ti in range(t - 1):
                for j in range(n):
                    alpha[ti + 1, j, li] = np.dot(alpha[ti, :, li], a_old[:, j]) * b_old[j, o[li, ti + 1]]

            # 计算beta
            for ti in range(t - 2, -1, -1):
                for j in range(n):
                    for i in range(n):
                        beta[ti, i, li] += a_old[i, j] * b_old[j, o[li, ti + 1]] * beta[ti + 1, j, li]

            # 计算gamma
            for ti in range(t):
                norm = np.dot(alpha[ti, :, li], beta[ti, :, li])
                gamma[ti, :, li] = alpha[ti, :, li] * beta[ti, :, li] / norm

            # 计算xi
            for ti in range(t - 1):
                norm = 0
                for i in range(n):
                    for j in range(n):
                        xi[ti, i, j, li] = alpha[ti, i, li] * a_old[i, j] * b_old[j, o[li, ti + 1]] * beta[
                            ti + 1, j, li]
                        norm += xi[ti, i, j, li]
                xi[ti, :, :, li] /= norm

        # 3、计算PI、A和B
        # 计算PI
        pi = np.sum(gamma[0, :, :], axis=1) / l
        # 计算A
        for i in range(n):
            for j in range(n):
                numerator = 0
                denominator = 0
                for li in range(l):
                    for ti in range(t - 1):
                        numerator += xi[ti, i, j, li]
                        denominator += gamma[ti, i, li]
                a[i, j] = numerator / denominator

        # 计算B
        for j in range(n):
            for mi in range(m):
                numerator = 0
                denominator = 0
                for li in range(l):
                    for ti in range(t):
                        if o[li, ti] == mi:
                            numerator += gamma[ti, j, li]
                        denominator += gamma[ti, j, li]
                b[j, mi] = numerator / denominator

        a_old = a
        b_old = b
        pi_old = pi

    return a, b, pi
