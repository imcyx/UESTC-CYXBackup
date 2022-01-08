# -*- coding: utf-8 -*-
"""
Bagging算法实现
弱分类器使用决策树桩

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import stump


def bagging_train(x, y, k=21):
    """
    完整Bagging算法实现
    输入
        x：数据集，y：标签，k：自助样本集数目
    输出
        weak_classifiers：弱分类器，agg_predict：聚集各弱分类器的预测
    """
    weak_classifiers = []
    n, d = x.shape
    w = np.array(np.ones(n) / n)        # 权重
    agg_predict = np.zeros(n)

    for it in range(k):
        # 有放回抽样
        sample_idx = np.random.choice(n, size=n, replace=True)
        best_stump, min_error, y_hat = stump.build_stump(x[sample_idx], y[sample_idx], w)
        weak_classifiers.append(best_stump)

        agg_predict += y_hat

    return weak_classifiers, agg_predict


def bagging_predict(x, bagging_model):
    """
    使用Bagging模型进行预测
    输入
        x：数据集，bagging_model：Bagging模型
    输出
        agg_predict：聚集各弱分类器的预测
    """
    n, d = x.shape
    predict_y = np.zeros(n)

    for i in bagging_model:
        pre_score = stump.stump_predict(x, i['dim'], i['thresh'], i['ineqal'])
        predict_y += pre_score
        # print("predict_y:", predict_y)
    agg_predict = np.sign(predict_y)
    return agg_predict
