# -*- coding: utf-8 -*-
"""
AdaBoost算法实现
弱分类器使用决策树桩

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import stump


def ada_boost_train(x, y, iters=40):
    """
    完整AdaBoost算法实现
    输入
        x：数据集，y：标签，iters：迭代次数
    输出
        weak_classifiers：弱分类器，agg_predict：聚集各弱分类器的预测
    """
    weak_classifiers = []
    n, d = x.shape
    w = np.array(np.ones(n) / n)        # 权重
    agg_predict = np.zeros(n)

    for it in range(iters):
        best_stump, epsilon, y_hat = stump.build_stump(x, y, w)
        if epsilon > 0.5:
            w = np.array(np.ones(n) / n)  # 重新设置权重
            break
        alpha = float(1 / 2 * np.log((1 - epsilon) / max(epsilon, 1e-16)))        # max(epsilon, 1e-16)防止零除错误
        best_stump['alpha'] = alpha
        weak_classifiers.append(best_stump)

        exponent = np.multiply(-1 * alpha * y.T, y_hat)
        w = np.multiply(w, np.exp(exponent))
        w = w / w.sum()     # 更新权重
        agg_predict += alpha * y_hat
        agg_errors = np.sign(agg_predict) != y.T
        agg_error_rate = np.sum(agg_errors) / n       # 加权错误率
        if agg_error_rate == 0.00:
            # 加权错误率为零，则没有必要再训练
            break

    return weak_classifiers, agg_predict


def ada_predict(x, ada_model):
    """
    使用AdaBoost模型进行预测
    输入
        x：数据集，ada_model：AdaBoost模型
    输出
        agg_predict：聚集各弱分类器的预测
    """
    n, d = x.shape
    predict_y = np.zeros(n)

    for i in ada_model:
        pre_score = stump.stump_predict(x, i['dim'], i['thresh'], i['ineqal'])
        predict_y += i['alpha'] * pre_score
        # print("predict_y:", predict_y)
    agg_predict = np.sign(predict_y)
    return agg_predict
