# -*- coding: utf-8 -*-
"""
HMM Viterbi算法实现
计算马尔科夫链最佳状态路径

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def hmm_viterbi(a, b, pi, o):
    """
    输入：
        a：转移概率矩阵
        b：发射概率矩阵
        pi：初始状态分布
        o：观测的输出序列
    输出：
        path：最佳状态路径
        p_star：该最佳状态路径概率
    """
    n = len(a)
    t = len(o)
    delta = np.zeros((t, n))
    psi = np.zeros((t, n))
    path = np.zeros(t, dtype=np.int)

    # 1、初始化
    delta[0, :] = pi * b[:, o[0]]
    psi[0, :] = 0

    # 2、归纳
    for ti in range(1, t):
        for j in range(n):
            tem = delta[ti - 1, :] * a[:, j]
            delta[ti, j] = np.max(tem) * b[j, o[ti]]
            psi[ti, j] = np.argmax(tem)

    # 3、终结
    p_star = np.max(delta[t - 1, :])
    path[t - 1] = np.argmax(delta[t - 1, :])

    # 4、路径回溯
    for ti in range(t - 2, -1, -1):
        path[ti] = psi[ti + 1, path[ti + 1]]

    return path, p_star
