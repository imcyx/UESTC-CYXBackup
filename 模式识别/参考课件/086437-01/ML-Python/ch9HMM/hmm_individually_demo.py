# -*- coding: utf-8 -*-
"""
HMM求取单个最有可能状态算法示例

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import hmm_individually


def main():
    a = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    b = np.array([[0.5, 0.5],
                  [0.6, 0.4],
                  [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])
    o = np.array([0, 1, 0, 1])

    path, gamma = hmm_individually.hmm_individually(a, b, pi, o)
    print(f'最佳状态路径：\n{path}')
    print(f'概率矩阵历史（t行是t时刻的gamma）：\n{gamma}')


if __name__ == "__main__":
    main()
