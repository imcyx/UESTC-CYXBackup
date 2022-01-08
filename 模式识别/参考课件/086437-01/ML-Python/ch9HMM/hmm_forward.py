# -*- coding: utf-8 -*-
"""
HMM前向算法实现
计算马尔科夫链观测状态的概率

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def hmm_forward(a, b, pi, o):
    """
    输入：
        a：转移概率矩阵
        b：发射概率矩阵
        pi：初始状态分布
        o：观测的输出序列
    输出：
        p：结果概率
        alpha：概率矩阵历史。t行是t时刻的alpha
    """
    n = len(a)
    t = len(o)
    alpha = np.zeros((t, n))

    # 1、初始化
    alpha[0, :] = pi * b[:, o[0]]

    # 2、归纳
    for i in range(t - 1):
        for j in range(n):
            # 计算sum(alpha_i(t-1)*a_ij)
            alpha[i + 1, j] = np.dot(alpha[i, :], a[:, j]) * b[j, o[i + 1]]

    # 3、终结
    p = np.sum(alpha[t - 1, :])

    return p, alpha
