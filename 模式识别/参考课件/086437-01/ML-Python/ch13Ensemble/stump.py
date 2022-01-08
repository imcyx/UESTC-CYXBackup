# -*- coding: utf-8 -*-
"""
Stump算法实现

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def stump_predict(x, dim, thresh, thresh_ineq):
    """
    决策树桩分类器预测
    输入
        x：数据集，dim：第几维，thresh：阈值，thresh_ineq：不等号，取值为"lt"（小于）或其他（大于）
    输出
        y_hat：预测结果
    """
    y_hat = np.ones(len(x))
    if thresh_ineq == "lt":
        y_hat[x[:, dim] <= thresh] = -1.0
    else:
        y_hat[x[:, dim] > thresh] = -1.0
    return y_hat


def build_stump(x, y, w):
    """
    构建决策树桩
    输入
        x：数据集，y：标签，w：权重
    输出
        best_stump：最优决策树桩，min_error：最小误差，y_hat：预测标签
    """
    n, d = x.shape
    num_steps = 10.0  # 每个特征的中间的可能值
    min_error = np.inf  # 最小错误率
    best_stump = {}  # 最优决策树桩分类器
    y_hat = np.zeros(n)  # 预测标签

    # 遍历每一维特征
    for di in range(d):
        min_val = x[:, di].min()
        max_val = x[:, di].max()
        step_size = (max_val - min_val) / num_steps  # 步长
        # 遍历第i维的可能值
        for si in range(-1, int(num_steps) + 1):
            thresh = min_val + float(si) * step_size  # 阈值
            for ineqal in ["lt", "gt"]:
                predict_y = stump_predict(x, di, thresh, ineqal)
                err = np.ones(n)  # 误差
                err[predict_y == y] = 0
                weighted_error = w @ err.T
                # print("dim ", di, ", thresh ", thresh, ", ineqal ", ineqal, ", weighted_error ", weighted_error)

                if weighted_error < min_error:
                    min_error = weighted_error
                    best_stump['dim'] = di
                    best_stump['thresh'] = thresh
                    best_stump['ineqal'] = ineqal
                    y_hat = predict_y.copy()
    return best_stump, min_error, y_hat
