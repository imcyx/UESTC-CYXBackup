# -*- coding: utf-8 -*-
"""
绘制鸢尾花数据集箱线图

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""
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
        x：属性，y：目标属性
    """
    with open(file, encoding="utf-8") as fr:
        fr.readline()  # 跳过标题行
        content = fr.readlines()
        x = [f.split(",")[: -1] for f in content]
        y = [f.split(",")[-1].strip("\n") for f in content]
    return x, y


def main():
    #  加载鸢尾花数据
    file_path = "../data/fisheriris.csv"
    x, y = read_csv(file_path)
    x = [float(f[0]) for f in x]
    s1 = x[0: 50]
    s2 = x[50: 100]
    s3 = x[100: 150]
    plt.figure()
    plt.boxplot(np.column_stack((s1, s2, s3)), labels=['setosa', 'versicolor', 'virginica'], notch=True, sym='r+')
    plt.show()


if __name__ == "__main__":
    main()
