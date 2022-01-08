# -*- coding: utf-8 -*-
"""
AdaBoost算法示例

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import adaboost


def load_data():
    """
    决策树桩分类器预测
    输入
        无
    输出
        data, labels：数据集和标签
    """
    data = np.array([[1.0, 2.1],
                    [2.0, 1.1],
                    [1.3, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0]])
    labels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return data, labels


def main():
    # 加载数据
    x, y = load_data()
    w = np.mat(np.ones(len(x)) / len(x))   # 权重
    best_stump, min_error, y_hat = adaboost.build_stump(x, y, w)
    print(best_stump)
    print(min_error)
    print(y_hat)

    iters = 9
    ada_classifier, _ = adaboost.ada_boost_train(x, y, iters)
    predict_label = adaboost.ada_predict(np.array([[5, 5], [0, 0]]), ada_classifier)
    print(predict_label)


if __name__ == "__main__":
    main()
