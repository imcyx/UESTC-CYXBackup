# -*- coding: utf-8 -*-
"""
使用基于内容（抽取物品特征）的协同过滤算法

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from scipy import optimize
import ml100k_utils


def normalize_ratings(y, r):
    """
    规范化评分矩阵
    预处理，减去平均评分，使每部电影的评分均值为0
    输入
        y：评分矩阵； r：是否已评分标志矩阵
    输出
        y_norm：正则化的评分矩阵； y_mean：评分均值
    """
    nu, nm = y.shape
    y_mean = np.zeros((1, nm))
    y_norm = np.zeros((nu, nm))
    for i in range(nm):
        # 查找评过分的影片
        idx = np.where(r[:, i] == 1)
        y_mean[0, i] = np.mean(y[idx, i])
        y_norm[idx, i] = y[idx, i] - y_mean[0, i]
    return y_norm, y_mean


def cost_func(params, y, r, nu, nm, nf, my_lambda):
    """
    协同过滤代价函数
    输入
        params：包含x和w部分
        y：评分矩阵，用户数 x 影片数，取值为1-5
        r：是否已经评分的矩阵，用户数 x 影片数，如果j用户对i影片评分，则R(i, j) = 1
        nu：用户数
        nm：影片数
        nf：特征数
        my_lambda：正则化因子
    输出
        j：代价
        grad：梯度
    """
    # 从params参数中抽取x和w
    # x为影片特征，nm x nf
    x = params[: nm * nf].reshape(nm, nf)
    # w为用户特征，nu x nf
    w = params[nm * nf:].reshape(nu, nf)

    # 计算协同过滤梯度
    # x_grad为对x求偏导数而得，nm x nf
    x_grad = np.dot(((np.dot(w, x.T) - y) * r).T, w) + my_lambda * x
    # w_grad为对w求偏导数而得，用户数 x 特征数
    w_grad = np.dot(((np.dot(w, x.T) - y) * r), x) + my_lambda * w

    # 计算正则化代价函数
    j = 0.5 * np.sum(np.square(r * (np.dot(w, x.T) - y)))
    reg_j = (my_lambda / 2) * (np.sum(np.square(w)) + np.sum(np.square(x)))
    j += reg_j

    grad = np.concatenate([x_grad.ravel(), w_grad.ravel()])

    return j, grad


def main():
    np.random.seed(123)
    # 加载电影评分数据集
    # 1、加载数据
    # 训练集
    # y_base为943x1682的矩阵，包含943个用户对1682影片的评分
    # r_base为943x1682的矩阵，r(i,j) = 1表示用户i对影片j评过分
    y_base, r_base = ml100k_utils.get_rating_matrix('../data/ml-100k/u.data')
    # 测试集
    y_test, r_test = ml100k_utils.get_rating_matrix('../data/ml-100k/u1.test')

    # 2、训练协同过滤算法模型

    # 规范化评分矩阵
    y_norm, y_mean = normalize_ratings(y_base, r_base)

    # 参数
    nu, nm = y_base.shape
    nf = 15     # 电影特征数
    my_lambda = 1  # 正则化参数

    # 随机设置初始化参数
    x = np.random.rand(nm, nf)
    w = np.random.rand(nu, nf)

    init_parms = np.concatenate([x.ravel(), w.ravel()])

    # 优化函数scipy.optimize.minimize的运行参数
    options = {'maxiter': 100}
    res = optimize.minimize(lambda t: cost_func(t, y_norm, r_base, nu, nm, nf, my_lambda),
                            init_parms,
                            method='TNC',
                            jac=True,
                            options=options)
    theta = res.x
    # 将返回的优化参数theta分离为x和w
    x = theta[:nm * nf].reshape(nm, nf)
    w = theta[nm * nf:].reshape(nu, nf)

    print('模型训练结束。\n')

    # 3、计算测试误差RMSE
    p = np.dot(w, x.T)
    pred = (p + np.tile(y_mean, (nu, 1))) * r_test

    rmse = np.sqrt(np.sum(np.square(y_test - pred)) / np.sum(r_test))
    print('RMSE：%f\n' % rmse)


if __name__ == "__main__":
    main()
