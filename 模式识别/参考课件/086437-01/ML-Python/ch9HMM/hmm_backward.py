# -*- coding: utf-8 -*-
"""
HMM后向算法实现

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def hmm_backward(a, b, pi, o):
    """
    输入：
        a：转移概率矩阵
        b：发射概率矩阵
        pi：初始状态分布
        o：观测的输出序列
    输出：
        p：结果概率
        beta：概率矩阵历史。t行是t时刻的beta
    """
    n = len(a)
    # m = len(b[0])
    t = len(o)
    beta = np.zeros((t, n))

    # 1、初始化
    beta[t - 1, :] = 1

    # 2、归纳
    for k in range(t - 2, -1, -1):
        for j in range(n):
            for i in range(n):
                beta[k, i] += a[i, j] * b[j, o[k + 1]] * beta[k + 1, j]

    # 3、终结
    p = np.sum(pi * beta[0, :] * b[:, o[0]])

    return p, beta
