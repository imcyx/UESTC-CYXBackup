# -*- coding: utf-8 -*-
"""
计算各种距离

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from scipy.spatial.distance import cdist
import warnings

# ignore by message
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")


def distance2(wy, wr, py, pr, metric='euclidean'):
    """
    对scipy.spatial.distance函数cdist()进行包装
    支持cdist()函数的各种距离参数，详见如下的metric参数说明
    输入
        wy：m x n 评分矩阵
        wr：m x n 是否评分矩阵。1为已评分，0为未评分
        py：1 x n 评分矩阵
        pr：1 x n 是否评分矩阵。1为已评分，0为未评分
        metric：距离参数，如：'cityblock'、'euclidean'、'mahalanobis'、'minkowski'、'cosine'、'correlation'、'jaccard'等，新增'pearson'
    输出
        dist：m x 1矩阵，第i个元素为Wy的第i行与Py的距离
    """
    m, n = wy.shape
    dist = np.zeros((m, 1))
    r = wr.astype(int) & np.tile(pr, (m, 1)).astype(int)
    for i in range(m):
        idx = np.where(r[i, :] == 1)
        if metric == 'pearson':
            # 处理两个向量没有重叠项
            if len(idx[0]) == 0:
                dist[i] = 0  # 不相关
            else:
                dist[i] = np.corrcoef(wy[i, idx], py[0, idx])[0][1]
            if np.isnan(dist[i]):
                dist[i] = 0
        else:
            if len(idx[0]) == 0:
                dist[i] = 10000  # 距离设为很大的数
            else:
                dist[i] = cdist(wy[i, idx], py[0, idx], metric)
    return dist
