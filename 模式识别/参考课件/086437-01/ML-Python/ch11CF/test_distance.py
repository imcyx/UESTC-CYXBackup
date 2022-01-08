# -*- coding: utf-8 -*-
"""
测试距离函数

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from scipy.spatial.distance import cdist
import distance


def main():
    # 测试甲乙丙三位用户与丁先生的欧氏距离
    wy = np.array([[5, 5], [5, 2], [4, 2]], dtype=int)
    wr = np.ones((3, 2), dtype=int)
    py = np.array([[4, 3]], dtype=int)
    pr = np.ones((1, 2), dtype=int)
    pd1 = cdist(wy, py)
    wpd1 = distance.distance2(wy, wr, py, pr, 'euclidean')
    print(f'三位用户与丁先生的欧氏距离cdist：\n{pd1}')
    print(f'三位用户与丁先生的欧氏距离distance2：\n{wpd1}')

    # 测试稀疏评分矩阵的欧氏距离
    # 以下测试中，注意到cdist和distance2的返回结果不一致，说明cdist没有考虑未评分情况
    wy2 = np.array([[3, 2, 0, 4, 5, 1, 2, 2], [2, 3, 4, 0, 2, 3, 0, 3]], dtype=int)
    wr2 = np.array([[1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 0, 1]], dtype=int)
    py2 = wy2[1].reshape(1, -1)
    pr2 = wr2[1].reshape(1, -1)
    pd2 = cdist(wy2, py2)
    wpd2 = distance.distance2(wy2, wr2, py2, pr2, 'euclidean')
    print(f'更为真实的评分例子的欧氏距离cdist：\n{pd2}')
    print(f'更为真实的评分例子的欧氏距离distance2：\n{wpd2}')

    # 测试余弦夹角距离
    pd3 = cdist(wy2, py2, 'cosine')
    wpd3 = distance.distance2(wy2, wr2, py2, pr2, 'cosine')
    print(f'测试余弦夹角距离cdist：\n{pd3}')
    print(f'测试余弦夹角距离distance2：\n{wpd3}')

    # 测试皮尔森相关系数
    pd4 = np.corrcoef(wy2, py2)[2][:2].reshape(-1, 1)
    wpd4 = distance.distance2(wy2, wr2, py2, pr2, 'pearson')
    print(f'测试皮尔森相关系数cdist：\n{pd4}')
    print(f'测试皮尔森相关系数distance2：\n{wpd4}')


if __name__ == "__main__":
    main()
