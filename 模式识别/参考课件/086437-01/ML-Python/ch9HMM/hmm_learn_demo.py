# -*- coding: utf-8 -*-
"""
HMM学习算法示例
假想的掷骰子游戏
两枚骰子，一枚正常，另一枚作弊
首先假定已知HMM参数，生成观测序列
然后根据观测序列估计HMM参数

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import hmm_learn
import hmm_generate


def main():
    t = 200
    l = 8
    a = np.array([[0.95, 0.05],
                  [0.10, 0.90]])
    b = np.array([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                  [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2]])
    pi = np.array([0.8, 0.2])
    o = np.zeros((l, t), dtype=np.int)
    for li in range(l):
        o[li, :], _ = hmm_generate.hmm_generate(a, b, pi, t)

    a_init = np.array([[0.6, 0.4],
                       [0.3, 0.7]])
    b_init = np.array([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                       [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 3 / 8]])
    pi_init = np.array([0.5, 0.5])

    a_est, b_est, pi_est = hmm_learn.hmm_learn(o, a_init, b_init, pi_init)

    print(f'真实的A：\n{a}')
    print(f'真实的B：\n{b}')
    print(f'真实的PI：\n{pi}')

    print(f'估计的A：\n{a_est}')
    print(f'估计的B：\n{b_est}')
    print(f'估计的PI：\n{pi_est}')


if __name__ == "__main__":
    main()
