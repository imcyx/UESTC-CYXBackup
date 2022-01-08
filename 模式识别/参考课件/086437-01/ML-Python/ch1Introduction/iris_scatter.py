# -*- coding: utf-8 -*-
"""
绘制鸢尾花数据集散点图和直方图

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


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


def plot_data(x_data1, x_data2):
    """
    可视化鸢尾花数据集的二维散点图
    输入
        x_data1, x_data2：要绘制的特征。假设三种类别各50个样本
    """
    # 分别获取三种鸢尾花的索引
    setosa = range(0, 50)
    versicolor = range(50, 100)
    virginica = range(100, 150)
    # setosa样本散点图
    plt.plot(x_data1[setosa], x_data2[setosa], 'ks', markerfacecolor='r', markersize=3)
    # versicolor样本散点图
    plt.plot(x_data1[versicolor], x_data2[versicolor], 'ko', markerfacecolor='g', markersize=3)
    # virginica样本散点图
    plt.plot(x_data1[virginica], x_data2[virginica], 'kd', markerfacecolor='b', markersize=3)


def main():
    #  加载鸢尾花数据
    file_path = "../data/fisheriris.csv"
    var_names: List[str] = ['sepal length', 'sepal width', 'petal length', 'petal width']
    x, y = read_csv(file_path)
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    plt.figure()
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i*4+j+1, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
            if i == j:
                plt.hist(x_data[:, i], 10)
            else:
                plot_data(x_data[:, i], x_data[:, j])

            if i == 0:
                plt.title(var_names[j], fontweight='normal', fontsize=9)
            if j == 0:
                plt.ylabel(var_names[i], rotation=90, fontsize=9)

    plt.show()


if __name__ == "__main__":
    main()
