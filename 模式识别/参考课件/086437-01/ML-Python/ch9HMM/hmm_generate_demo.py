# -*- coding: utf-8 -*-
"""
假想的掷骰子游戏
两枚骰子，一枚正常，另一枚作弊

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import hmm_generate


def main():
    a = np.array([[0.95, 0.05],
                  [0.10, 0.90]])
    b = np.array([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                  [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])
    pi = np.array([0.5, 0.5])

    seq, states = hmm_generate.hmm_generate(a, b, pi, 100)
    print(f'观测序列O：\n{seq + 1}')
    print(f'隐藏序列Q：\n{states}')


if __name__ == "__main__":
    main()
