# -*- coding: utf-8 -*-
"""
绘制tanh（双曲正切函数）函数的导数

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import matplotlib.pyplot as plt
import numpy as np


def derivative_tanh(z):
    f = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    d = 1 - np.square(f)
    return d


def main():
    # x取值，start到stop范围内间隔均匀的100个数
    x_vals = np.linspace(start=-10., stop=10., num=100)

    # Tanh激活函数的导数
    print(derivative_tanh(np.array([-1., 0., 1.])))
    y_vals = derivative_tanh(x_vals)

    # 绘制Tanh激活函数图像
    plt.plot(x_vals, y_vals, "r--", label="Derivative Tanh", linewidth=1.5)
    plt.ylim([-0.5, 1.5])
    plt.xlabel('z')
    plt.ylabel("f'(z)")
    plt.legend(loc="upper left")
    plt.grid(axis="y")
    plt.show()


if __name__ == "__main__":
    main()
