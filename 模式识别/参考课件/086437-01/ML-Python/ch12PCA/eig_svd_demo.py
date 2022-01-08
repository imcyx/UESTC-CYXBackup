# -*- coding: utf-8 -*-
"""
测试本征值分解和奇异值分解

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def main():
    np.random.seed(1234)
    print('******   本征值分解   ******\n')
    a = np.random.rand(4, 4)
    print(f'a: {a}\n')

    print('w, v = np.linalg.eig(a)\n计算a的本征值和本征向量\n')
    w, v = np.linalg.eig(a)
    print(f'w: {w}\n')
    print(f'v: {v}\n')

    print('验证 a*v = w*v\n')
    av = np.dot(a, v)
    print(f'a*v: {av}\n')
    w_diag = np.diag(w)
    wv = np.dot(v, w_diag)
    print(f'w*v: {wv}\n')

    print('******   奇异值分解   ******\n')
    a = np.random.rand(4, 3)
    print(f'a: {a}\n')

    print('u, s, vh = numpy.linalg.svd(a)\n计算a的奇异值分解\n')
    u, s, vh = np.linalg.svd(a)
    print(f'u: {u}\n')
    print(f's: {s}\n')
    print(f'vh: {vh}\n')

    print('验证 a = u*s*vh\n')
    s_diag = np.diag(s)
    s_diag = np.row_stack((s_diag, np.zeros((1, 3))))
    usvh = np.dot(np.dot(u, s_diag), vh)
    print(f'a: {a}\n')
    print(f'u*s*vh: {usvh}\n')


if __name__ == "__main__":
    main()
