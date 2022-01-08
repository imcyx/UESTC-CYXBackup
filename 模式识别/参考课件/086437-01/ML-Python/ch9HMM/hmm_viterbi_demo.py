# -*- coding: utf-8 -*-
"""
掷硬币实验
三个硬币，一个正常币，2个偏币

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import hmm_viterbi


def main():
    a = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    b = np.array([[0.5, 0.5],
                  [0.8, 0.2],
                  [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])
    o = np.array([0, 1, 0, 1])

    path, p_star = hmm_viterbi.hmm_viterbi(a, b, pi, o)
    print(f'最佳状态路径：\n{path}')
    print(f'该最佳状态路径概率：\n{p_star}')


if __name__ == "__main__":
    main()
