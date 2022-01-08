# -*- coding: utf-8 -*-
"""
给定HMM模型，生成HMM观测系列

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def sampling(prob_distribution):
    """
     输入
        probDistribution ： 概率分布
    输出
        idx ： 抽样到的类别索引
    """
    idx = np.random.choice(len(prob_distribution), p=prob_distribution)
    return idx


def hmm_generate(a, b, pi, t):
    """
    生成HMM观测系列
    输入：
        a： 转移概率矩阵
        b： 发射概率矩阵
        pi： 初始状态分布
        t： 序列长度
    输出：
        seq： 观测序列
        states： 状态序列
    """
    epsilon = 1e-10  # 最小舍入误差

    # 必要的输入数据检查
    if np.sum(pi - abs(pi)) != 0 or np.sum(a - np.abs(a)) != 0 or np.sum(b - np.abs(b)) != 0:
        print('输入矩阵错误，每个元素必须大于等于0。')
    if np.abs(np.sum(pi) - 1) > epsilon or \
            np.any(np.abs(np.sum(a, axis=1, keepdims=True) - 1) > epsilon) or \
            np.any(np.abs(np.sum(b, axis=1, keepdims=True) - 1) > epsilon):
        print('输入矩阵错误，每行元素累加和必须为1。')

    # 1、初始化
    seq = np.zeros(t, dtype=np.int)
    states = np.zeros(t, dtype=np.int)
    qi = sampling(pi)  # 当前状态值

    # 2、迭代输出
    for i in range(t):
        states[i] = qi
        seq[i] = sampling(b[qi, :])
        qi = sampling(a[qi, :])

    return seq, states

