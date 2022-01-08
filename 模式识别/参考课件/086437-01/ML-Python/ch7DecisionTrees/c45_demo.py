# -*- coding: utf-8 -*-
"""
C4.5决策树算法示例

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import c45
import sys

sys.path.append('..')
from plot_tree import create_plot

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
        data = np.mat([list(line.strip("\n").split(delimiter)) for line in content])
    return data


def main():
    # 加载天气数据
    data = load_dataset("../data/weather_numeric.csv")
    feature_list = ['outlook', 'temperature', 'humidity', 'windy']
    print(data.shape)
    print(set(data[:, -1].T.tolist()[0]))
    # 构建决策树
    my_tree = c45.build_c45_tree(data, feature_list)
    print(f"构建的C4.5决策树描述：\n{my_tree}\n")
    # 绘制决策树
    create_plot(my_tree)
    # 决策树分类
    y_hat = []
    for i in range(len(data)):
        y_hat.append(c45.classify(my_tree, feature_list, data[i, :-1]))
    print('天气问题结果：')
    print('训练集上的分类正确率： {:.2%}'.format(np.mean(y_hat == data[:, -1].squeeze())))

    # # 加载鸢尾花数据
    # data = load_dataset("../data/fisheriris.csv")
    # data = data[1:, :]
    # feature_list = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # # 构建决策树
    # my_tree = c45.build_c45_tree(data, feature_list)
    # print(f"构建的C4.5决策树描述：\n{my_tree}\n")
    # # 绘制决策树
    # create_plot(my_tree)
    # # 决策树分类
    # y_hat = []
    # for i in range(len(data)):
    #     y_hat.append(c45.classify(my_tree, feature_list, data[i, :-1]))
    # print('鸢尾花问题结果：')
    # print('训练集上的分类正确率： {:.2%}'.format(np.mean(y_hat == data[:, -1].squeeze())))


if __name__ == "__main__":
    main()
