# -*- coding: utf-8 -*-
"""
随机森林示例

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from decision_tree_model import ClassificationTree
from random_forest_model import RandomForest


def load_dataset(file_name, delimiter=','):
    """
    加载CSV或TSV数据集
    输入参数
        file_name：CSV文件名，delimiter：分隔符
    输出参数
        data：数据集
    """
    with open(file_name) as file:
        content = file.readlines()
        data = np.array([list(line.strip("\n").split(delimiter)) for line in content])
    return data


def shuffle_data(x, y, seed=None):
    """ 随机置乱数据集顺序 """
    if seed:
        np.random.seed(seed)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    return x[idx], y[idx]


def train_test_split(x, y, test_size=0.4, shuffle=True, seed=None):
    """ 划分训练集和测试集 """
    if shuffle:
        x, y = shuffle_data(x, y, seed)
    # 根据划分比例计算划分点
    split_point = int(len(y) * (1 - test_size))
    x_train, x_test = x[:split_point], x[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    return x_train, x_test, y_train, y_test


def main():
    # 加载鸢尾花数据
    data = load_dataset("../data/fisheriris.csv")
    data = data[1:, :]      # 去掉标题
    x_data = data[:, :-1]
    # 转换为Numpy数组
    x = np.zeros((len(x_data), len(x_data[0])))
    for i in range(len(x[0])):
        x[:, i] = [float(f[i]) for f in x_data]

    # 目标setosa为0，versicolor为1，virginica为2
    y = np.row_stack((np.zeros((50, 1)), np.ones((50, 1)), np.ones((50, 1)) * 2)).reshape(-1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, seed=2)
    print("训练集数据形状：", x_train.shape)
    print("训练集标签形状：", y_train.shape)

    dt_clf = ClassificationTree()
    dt_clf.fit(x_train, y_train)
    y_pred = dt_clf.predict(x_test)
    print('决策树算法在测试集上的分类正确率：{:.2%}'.format(np.mean(y_test == y_pred)))

    rf_clf = RandomForest(n_tree=90)
    rf_clf.fit(x_train, y_train)
    y_pred = rf_clf.predict(x_test)
    print('随机森林算法在测试集上的分类正确率：{:.2%}'.format(np.mean(y_test == y_pred)))


if __name__ == "__main__":
    main()
