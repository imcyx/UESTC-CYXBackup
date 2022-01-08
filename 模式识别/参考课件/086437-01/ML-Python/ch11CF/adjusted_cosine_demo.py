# -*- coding: utf-8 -*-
"""
基于物品的协同过滤算法示例
使用调整余弦相似度计算两个物品间的相似度并进行预测

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import distance


def adj_cosine(ya, yb, ra, rb, avg):
    """
    求两个列向量的调整余弦相似度
    输入
        ya, yb：评分列向量
        ra, rb：是否评分标志列向量
        avg：用户评分均值列向量
    输出
        sim：两个列向量的调整余弦相似度
    """
    sim = 0
    r = ra & rb
    idx = np.where(r[:, 0] == 1)
    if np.sum(idx) == 0:
        return sim

    a = ya - avg
    b = yb - avg
    a = a[idx]
    b = b[idx]
    sim = np.sum(a * b) / np.sqrt(np.sum(a * a)) / np.sqrt(np.sum(b * b))
    return sim


def main():
    # 1、直接计算相似度矩阵
    # 评分矩阵
    y = np.array([[0, 4, 5, 4], [0, 2, 3, 3], [4, 3, 0, 2], [3, 4, 4, 2], [5, 4, 5, 0]], dtype=int)
    r = np.array([[0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=int)
    print(f'\n评分矩阵：\n{y}')

    nu, nm = y.shape
    avg_user = np.zeros((nu, 1))
    for i in range(nu):
        avg_user[i, 0] = np.mean(y[i, np.where(r[i, :] == 1)])

    # 相似度矩阵
    sim = np.zeros((nm, nm))

    for i in range(nm - 1):
        for j in range(i + 1, nm):
            sim[i, j] = adj_cosine(y[:, i].reshape(-1, 1), y[:, j].reshape(-1, 1),
                                   r[:, i].reshape(-1, 1), r[:, j].reshape(-1, 1), avg_user)
            sim[j, i] = sim[i, j]  # 对称矩阵

    print(f'\n相似度矩阵：\n{sim}')

    # 预测用户甲对物品A的评分
    pred = np.sum(sim[0, :] * y[0, :]) / np.sum(abs(sim[0, :]))
    print('\n直接计算得到用户甲对物品A的评分：%f\n有点不靠谱！\n' % pred)

    # 2、先规范化，再计算相似度矩阵
    min_y = 1
    max_y = 5
    ny = (2 * (y - min_y) - (max_y - min_y)) / (max_y - min_y)
    ny = ny * r

    print(f'\n规范化后的评分矩阵：\n{ny}')

    n_avg_user = np.zeros((nu, 1))
    for i in range(nu):
        n_avg_user[i, 0] = np.mean(ny[i, np.where(r[i, :] == 1)])

    # 相似度矩阵
    n_sim = np.zeros((nm, nm))

    for i in range(nm - 1):
        for j in range(i + 1, nm):
            n_sim[i, j] = adj_cosine(ny[:, i].reshape(-1, 1), ny[:, j].reshape(-1, 1),
                                     r[:, i].reshape(-1, 1), r[:, j].reshape(-1, 1), n_avg_user)
            n_sim[j, i] = n_sim[i, j]  # 对称矩阵

    print(f'\n规范化后的相似度矩阵：\n{n_sim}')

    # 预测用户甲对物品A的评分
    n_pred = np.sum(n_sim[0, :] * ny[0, :]) / np.sum(abs(n_sim[0, :]))
    print('\n规范化后直接预测用户甲对物品A的评分：%f\n' % n_pred)
    # 如果经过规范化，必须逆运算回来
    n_pred = 0.5 * ((n_pred + 1) * (max_y - min_y)) + min_y
    print('\n规范化后逆运算计算得到的用户甲对物品A的评分：%f\n好像好多了！\n' % n_pred)


if __name__ == "__main__":
    main()
