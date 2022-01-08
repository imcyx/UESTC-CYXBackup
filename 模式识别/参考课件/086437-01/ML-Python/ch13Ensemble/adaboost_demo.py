# -*- coding: utf-8 -*-
"""
AdaBoost算法示例

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import adaboost
import stump


def read_csv(file):
    """
    读取CSV文件，允许有多个属性，文件最后一列为目标属性
    输入参数：
        file：文件名
    返回：
        x_data：属性，y：目标属性
    """
    with open(file, encoding="utf-8") as fr:
        fr.readline()  # 跳过标题行
        content = fr.readlines()
        x_data = [f.split(",")[: -1] for f in content]
        y = [f.split(",")[-1].strip("\n") for f in content]
    return x_data, y


def main():
    np.random.seed(5)
    #  加载鸢尾花数据
    file_path = "../data/fisheriris.csv"
    x, _ = read_csv(file_path)

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    # 仅使用versicolor和virginica两类
    x_data = x_data[50: 150, :]
    # 目标versicolor为-1，virginica为1
    y_data = np.concatenate((-1 * np.ones(50), np.ones(50)))

    # 划分训练集测试集
    # 随机置乱
    idx = np.random.permutation(len(x_data))
    x_data = x_data[idx]
    y_data = y_data[idx]
    train_split = int(len(x_data) * 0.7)
    x_train = x_data[:train_split]
    y_train = y_data[:train_split]
    x_test = x_data[train_split:]
    y_test = y_data[train_split:]

    w = np.array(np.ones(len(x_train)) / len(x_train))   # 权重
    best_stump, min_error, y_hat = stump.build_stump(x_train, y_train, w)
    print('决策树桩算法在训练集上的分类正确率：{:.2%}'.format(np.mean(y_hat == y_train)))
    predict_y = stump.stump_predict(x_test, best_stump['dim'], best_stump['thresh'], best_stump['ineqal'])
    print('决策树桩算法在测试集上的分类正确率：{:.2%}\n'.format(np.mean(predict_y == y_test)))

    iters = 10      # 可以修改此参数，看效果
    ada_classifier, _ = adaboost.ada_boost_train(x_train, y_train, iters)
    y_hat = adaboost.ada_predict(x_train, ada_classifier)
    print('AdaBoost算法在训练集上的分类正确率：{:.2%}'.format(np.mean(y_hat == y_train)))
    predict_y = adaboost.ada_predict(x_test, ada_classifier)
    print('AdaBoost算法在测试集上的分类正确率：{:.2%}'.format(np.mean(predict_y == y_test)))


if __name__ == "__main__":
    main()
