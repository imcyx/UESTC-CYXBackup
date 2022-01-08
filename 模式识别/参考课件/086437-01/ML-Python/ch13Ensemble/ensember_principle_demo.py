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
    weak_classifiers = 25       # 弱分类器数
    err = np.linspace(0, 1, 100)  # 弱分类器误差
    err_ensember = np.zeros_like(err)

    # 计算集成分类器误差
    for ei in range(len(err)):
        for i in range(int(np.ceil(weak_classifiers / 2)), weak_classifiers + 1):
            err_ensember[ei] += comb(weak_classifiers, i) * np.power(err[ei], i) * np.power(1 - err[ei], weak_classifiers - i)

    # 绘制误差曲线
    plt.figure()
    plt.plot(err, err, 'b--', label=u'弱分类器')
    plt.plot(err, err_ensember, 'r-', linewidth=2, label=u'集成分类器')

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel(u'基分类器误差')
    plt.ylabel(u'集成分类器误差')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
