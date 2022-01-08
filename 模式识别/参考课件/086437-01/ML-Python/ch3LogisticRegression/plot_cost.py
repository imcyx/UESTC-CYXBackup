"""
绘制代价函数

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import matplotlib.pyplot as plt
import numpy as np


def main():
    #   生成0~1之间的100个数
    h = np.linspace(0, 1, num=100, endpoint=False)
    h = np.delete(h, 0, axis=0)     # 丢弃0，以免log0没有定义
    cost1 = -np.log(h)
    cost0 = -np.log(1 - h)

    # y=1
    plt.figure()
    plt.plot(h, cost1)
    plt.xlabel(r'h(x;$\theta$)')
    plt.ylabel(r'cost(h(x;$\theta$),y)')
    plt.title('y=1')
    plt.xticks(np.linspace(0, 1, 2))
    plt.yticks(np.linspace(0, 5, 6))
    plt.show()

    # y=0
    plt.figure()
    plt.plot(h, cost0)
    plt.xlabel(r'h(x;$\theta$)')
    plt.ylabel(r'cost(h(x;$\theta$),y)')
    plt.title('y=0')
    plt.xticks(np.linspace(0, 1, 2))
    plt.yticks(np.linspace(0, 5, 6))
    plt.show()


if __name__ == "__main__":
    main()
