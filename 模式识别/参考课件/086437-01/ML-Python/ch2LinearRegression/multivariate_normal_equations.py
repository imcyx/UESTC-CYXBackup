# -*- coding: utf-8 -*-
"""
多变量线性回归
使用正规方程，线性拟合UCI数据集——Relative CPU Performance Data

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def read_csv(file):
    """
    读取CSV文件，允许有多个属性，文件最后一列为目标属性
    输入参数：
        file：文件名
    返回：
        x：属性，y：目标属性
    """
    with open(file, encoding="utf-8") as fr:
        content = fr.readlines()
        x = [f.split(",")[: -1] for f in content]
        y = [float(f.split(",")[-1].strip("\n")) for f in content]
    return x, y


def normal_equation(x, y):
    """
    正规方程求解theta参数集
    输入
        x：特征矩阵，y：目标属性
    输出
        theta：参数集，向量
    """
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    return theta


def main():
    # 加载CPU数据
    file_path = "../data/machine.csv"
    x, y = read_csv(file_path)

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x[0])):
        x_data[:, i] = [float(f[i]) for f in x]
    y_data = np.array(y).reshape(-1, 1)

    n = len(y_data)  # 样本数

    # 添加一列全1，以扩展x
    x_data = np.column_stack((np.ones((n, 1)), x_data))

    # 用正规方程计算theta参数
    theta = normal_equation(x_data, y_data)

    # 打印找到的参数
    print(f'正规方程结果，Theta参数：\n {theta}')

    # 计算误差
    estimate_y = np.dot(x_data, theta)
    err = y_data - estimate_y
    rmse = np.squeeze(np.sqrt(np.dot(err.T, err) / n))
    print(f'\nRMSE：  {rmse} \n')


if __name__ == "__main__":
    main()
