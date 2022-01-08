# -*- coding: utf-8 -*-
"""
绘制Sigmoid激活函数

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def main():
    # x取值，start到stop范围内间隔均匀的100个数
    x_vals = np.linspace(start=-10., stop=10., num=100)

    # Sigmoid激活函数
    print(sigmoid(np.array([-1., 0., 1.])))
    y_vals = sigmoid(x_vals)

    # 绘制Sigmoid激活函数图像
    plt.plot(x_vals, y_vals, "r--", label="Sigmoid", linewidth=1.5)
    plt.ylim([-0.5, 1.5])
    plt.xlabel('z')
    plt.ylabel('g(z)')
    plt.legend(loc="upper left")
    plt.grid(axis="y")
    plt.show()


if __name__ == "__main__":
    main()
