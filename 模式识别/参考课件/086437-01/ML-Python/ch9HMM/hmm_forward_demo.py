# -*- coding: utf-8 -*-
"""
HMM前向算法示例
掷硬币实验模拟
三个硬币，一个正常币，2个偏币

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import hmm_forward


def main():
    a = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    b = np.array([[0.5, 0.5],
                  [0.8, 0.2],
                  [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])
    o = np.array([0, 1, 0, 1])

    p, alpha = hmm_forward.hmm_forward(a, b, pi, o)
    print(f'结果概率：\n{p}')
    print(f'概率矩阵历史（t行是t时刻的alpha）：\n{alpha}')


if __name__ == "__main__":
    main()
