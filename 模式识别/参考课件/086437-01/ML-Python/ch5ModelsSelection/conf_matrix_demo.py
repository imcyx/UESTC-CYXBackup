# -*- coding: utf-8 -*-
"""
混淆矩阵示例

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import one_vs_one_function


def main():
    # 返回鸢尾花数据集的真实标签和预测标签
    y_real, y_hat = one_vs_one_function.iris_classifier()
    print('训练集上的分类正确率：: {:.2%}'.format(np.mean(y_hat == y_real)))
    # 混淆矩阵
    conf_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            y_real_i = y_real[:, 0] == i
            y_hat_j = y_hat[:, 0] == j
            conf_matrix[i, j] = np.sum(y_real_i * y_hat_j)

    print('\n混淆矩阵:')
    print(conf_matrix)


if __name__ == "__main__":
    main()
