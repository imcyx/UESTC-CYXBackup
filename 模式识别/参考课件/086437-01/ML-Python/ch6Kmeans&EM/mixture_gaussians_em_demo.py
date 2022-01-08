# -*- coding: utf-8 -*-
"""
使用EM算法求解混合高斯问题

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


def sampling(prob_distribution):
    """
     输入
        probDistribution ： 概率分布
    输出
        idx ： 抽样到的类别索引
    """
    idx = np.random.choice(len(prob_distribution), p=prob_distribution)
    return idx


def plot_mixture_gaussians(x, mu, sigma, iter, q):
    """
    绘制当前迭代中的状况
    输入参数
        x：输入，mu：均值，sigma：协方差，iter：当前迭代次数，q：概率分布
    输出参数
        无
    """
    plt.clf()
    left, right = -3, 6
    bottom, top = -6, 6
    n = len(x)

    # 使用q作为RGB颜色值
    # 如果K不为3，此处需修改
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=q)

    # 绘制高斯轮廓图
    # 网格范围
    u = np.linspace(left, right, 150)
    v = np.linspace(bottom, top, 150)
    uu, vv = np.meshgrid(u, v)  # 生成网格数据
    uv = np.column_stack((uu.ravel(), vv.ravel()))
    k = len(mu)
    for ki in range(k):
        sqrt_det_sigma = np.sqrt(np.linalg.det(sigma[ki]))
        inv_sigma = np.linalg.pinv(sigma[ki])
        head = 1 / (2 * np.pi * sqrt_det_sigma)
        z = np.zeros(len(uv))
        for i in range(len(uv)):
            z[i] = head * np.exp(-1 / 2 * ((uv[i] - mu[ki]) @ inv_sigma @ (uv[i] - mu[ki]).T))
        # 保持维度一致
        z = z.reshape(uu.shape)
        # 画图
        plt.contour(uu, vv, z)

    # 坐标
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((left, right))
    plt.ylim((bottom, top))
    plt.title(f'第{iter + 1}次迭代')
    plt.pause(3)


def plot_lower_bound_history(b):
    """
    绘制下界历史
    输入参数
        b：下界历史
    输出参数
        无
    """
    plt.figure()
    plt.plot(np.arange(1, len(b)), b[1:], 'k-')
    plt.xlabel('迭代次数')
    plt.ylabel('下界')
    plt.show()


def main():
    # 随机生成数据
    np.random.seed(1234)
    # 三个高斯
    real_means = np.array([[0.0, -3.2], [1.0, 3.5], [3.0, -1.0]])
    real_covs = np.zeros((3, 2, 2))
    real_covs[0] = np.array([[0.58, -0.05], [-0.05, 1.55]])
    real_covs[1] = np.array([[0.65, -0.15], [-0.15, 1.12]])
    real_covs[2] = np.array([[0.80, -0.05], [-0.05, 0.80]])
    real_priors = np.array([0.3, 0.3, 0.4])

    # 一次生成一个数据点
    n = 100      # 数据点总数
    x = np.zeros((n, 2))
    # 选择一个高斯，并随机抽样数据
    for i in range(n):
        comp = sampling(real_priors)
        x[i] = np.random.multivariate_normal(real_means[comp], real_covs[comp], 1)

    # 初始化混合高斯参数
    k = 3
    # 均值矩阵。注意每个高斯的均值是行向量，与数据集一致
    means = np.random.randn(k, 2)
    covs = np.zeros((k, 2, 2))      # 预分配空间
    for ki in range(k):
        covs[ki] = np.random.rand() * np.eye(2)
    priors = np.tile(1 / k, k)       # 均匀分布

    # 设置EM算法参数
    max_its = 100        # 最大迭代次数
    q = np.zeros((n, k))
    n, d = x.shape
    plot_points = np.concatenate((np.arange(1, 10, 2), np.arange(14, 30, 5), np.arange(39, max_its, 10)))
    b = np.zeros(max_its)
    b[0] = -np.inf     # 下界初值
    tol = 1e-2
    eps = 1e-16

    # 运行EM算法
    tmp = np.zeros((n, k))
    plt.figure()
    # 打开交互模式
    plt.ion()
    for iter in range(max_its):
        # 更新q
        for ki in range(k):
            # 下界B的常数部分
            const = -(d / 2) * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(covs[ki]))
            x_mu = x - means[ki]      # X - mu
            tmp[:, ki] = const - 0.5 * np.diag(x_mu @ np.linalg.inv(covs[ki]) @ x_mu.T)    # 注意这里的Xmu是行向量

        # 计算似然的下界
        if iter > 0:
            # 按照公式计算下界B
            b[iter] = np.sum(q * np.log(np.tile(priors, (n, 1)))) + np.sum(q * tmp) - np.sum(q * np.log(q))
            # 是否已经收敛？
            if np.abs(b[iter] - b[iter - 1]) < tol:
                break

        tmp += np.tile(priors, (n, 1))
        q = np.exp(tmp - np.tile(np.max(tmp, axis=1, keepdims=True), (1, k)))

        # 避免数值计算问题
        q[q < eps] = eps
        q[q > 1 - eps] = 1
        q = np.true_divide(q, np.tile(np.sum(q, axis=1, keepdims=True), (1, k)))
        # 更新priors
        priors = np.mean(q, axis=0)
        # 更新均值means
        for ki in range(k):
            means[ki] = np.true_divide(np.sum(x * np.tile(q[:, ki].reshape(-1, 1), (1, d)), axis=0), np.sum(q[:, ki]))
        # 更新协方差covs
        for ki in range(k):
            x_mu = x - np.tile(means[ki], (n, 1))
            covs[ki] = np.true_divide((x_mu * np.tile(q[:, ki].reshape(-1, 1), (1, d))).T @ x_mu, np.sum(q[:, ki]))

        if i in plot_points:
            # 绘制
            plot_mixture_gaussians(x, means, real_covs, iter, q)

    plot_mixture_gaussians(x, means, real_covs, iter, q)
    print('EM算法运行完毕！')

    # 关闭交互模式
    plt.ioff()
    plt.show()

    # 绘制下界历史
    plot_lower_bound_history(b)

    print('比较真实参数与EM估计的参数\n')
    print(f'真实的均值：\n{real_means}\n')
    print(f'真实的协方差：\n{real_covs}\n')
    print(f'真实的先验：\n{real_priors}\n')
    print(f'估计的均值：\n{means}\n')
    print(f'估计的协方差：\n{covs}\n')
    print(f'估计的先验：\n{priors}\n')


if __name__ == "__main__":
    main()
