# -*- coding: utf-8 -*-
"""
基分类器和集成分类器误差分析

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def main():
    eps = 1e-15
    epsilon = np.linspace(0 + eps, 1 - eps, 100)
    alpha = 1 / 2 * np.log((1 - epsilon) / epsilon)

    # 绘制alpha曲线
    plt.figure()
    plt.plot(epsilon, alpha, 'r-', linewidth=1, label=r'$\alpha$')

    plt.xlim((0, 1))
    plt.ylim((-5, 5))
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$ \alpha= \frac{1}{2} log\left(\frac{1 - \epsilon} {\epsilon} \right)$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
