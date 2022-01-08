# -*- coding: utf-8 -*-
"""
基于物品的协同过滤算法示例
使用加权的Slope One算法

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
# import ml100k_utils


def main():
    # 评分上下界
    max_y = 5
    min_y = 1

    # 书上的例子
    y_base = np.array([[4, 3, 5], [5, 2, 0], [0, 3, 4], [5, 0, 3]], dtype=np.int)
    r_base = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int)
    y_test = np.array([[0, 0, 0], [0, 0, 4], [5, 0, 0], [0, 2, 0]], dtype=np.int)
    r_test = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int)

    users, items = y_base.shape

    # # MovieLens 100k数据集
    # # 训练集
    # y_base, r_base = ml100k_utils.get_rating_matrix('../data/ml-100k/u1.base')
    # # 测试集
    # y_test, r_test = ml100k_utils.get_rating_matrix('../data/ml-100k/u1.test')
    # users, items, _ = ml100k_utils.get_uinfo()

    # 计算差值
    dev = np.zeros((items, items))
    for i in range(items - 1):
        for j in range(i + 1, items):
            r = r_base[:, i] & r_base[:, j]     # 都评分的标志

            if np.sum(r) == 0:
                # 没有任何用户同时评分
                dev[i, j] = np.nan
                dev[j, i] = np.nan
            else:
                dev[i, j] = np.sum(y_base[:, i] * r - y_base[:, j] * r) / np.sum(r)
                dev[j, i] = - dev[i, j]

    print('差值矩阵：\n')
    print(dev)

    # 使用加权的Slope One算法进行预测
    pws1 = np.zeros((users, items))
    # 遍历所有用户
    for u in range(users):
        for j in iter(np.array(np.where(r_test[u, :] == 1))[0]):
            numerator = 0       # 分子
            denominator = 0     # 分母

            for i in iter(np.array(np.where(r_base[u, :] == 1))[0]):
                # 实现i∈S(u)-{j}
                if i == j:
                    continue
                cji = np.sum(r_base[:, j] & r_base[:, i])
                # 跳过没有任何用户同时评分的物品对
                if cji != 0:
                    numerator += (dev[j, i] + y_base[u, i]) * cji
                    denominator += cji

            if denominator != 0:
                pws1[u, j] = numerator / denominator
                # 检查预测评分范围，要在上下界之间
                if pws1[u, j] > max_y:
                    pws1[u, j] = max_y
                if pws1[u, j] < min_y:
                    pws1[u, j] = min_y
            else:
                pws1[u, j] = np.sum(y_base[u, :]) / np.sum(r_base[u, :])    # 用该用户的平均评分作为预测值
                print('无法计算推荐pws1[%d, %d]\n' % (u, j))
    # 计算RMSE
    print('预测评分：\n')
    print(pws1)

    rmse = np.sqrt(np.sum(np.square(y_test - pws1)) / np.sum(r_test))
    print('RMSE：%f\n' % rmse)


if __name__ == "__main__":
    main()
