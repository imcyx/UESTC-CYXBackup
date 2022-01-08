# -*- coding: utf-8 -*-
"""
QP SVM原问题分类
这里只测试二元分类，使用二次规划

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from cvxopt import matrix, solvers


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


def predict(w, b, x):
    """
    使用学习到的参数theta来预测x的标签
    """
    pred = np.ones((len(x), 1))
    # 如果w'*x + b >= 0，预测为1，否则为-1
    p = (np.dot(x, w) + b >= 0).astype(np.int)
    neg = np.where(p == 0)
    pred[neg, 0] = -1
    return pred


def main():
    # 加载数据
    file_path = "../data/fisheriris.csv"
    x, _ = read_csv(file_path)
    x = x[:100]

    # 转换为Numpy数组
    x_data = np.zeros((len(x), len(x[0])))
    for i in range(len(x_data[0])):
        x_data[:, i] = [float(f[i]) for f in x]

    # 目标setosa为-1，versicolor为1
    y_data = np.row_stack((-1 * np.ones((50, 1)), np.ones((50, 1))))

    # 调用QP计算参数
    # 数据矩阵大小
    n, d = x_data.shape

    # 调用QP前的准备
    H = np.dot(y_data, y_data.T) * np.dot(x_data, x_data.T)
    H = matrix(H)
    f = matrix(- np.ones((n, 1)))
    ceq = matrix(np.zeros(1))
    Aeq = matrix(y_data.T)
    A = matrix(- np.eye(n))
    c = matrix(np.zeros((n, 1)))

    # 调用QP函数
    sol = solvers.qp(H, f, A, c, Aeq, ceq)
    alpha = np.array(sol['x'])
    print(f'最优alpha：\n{alpha}')

    # 由alpha求解w和b
    w = np.zeros((d, 1))
    for i in range(n):
        w += alpha[i] * y_data[i, 0] * x_data[i].reshape(-1, 1)

    # 计算支持向量的索引
    idx = np.where(alpha[:, 0] > 0.00001)[0]
    print('\n一共有 %d 个支持向量 \n\n' % len(idx))
    b = y_data[idx[0], 0] - np.dot(w.T, x_data[idx[0]].T)

    # 显示w和b参数
    print(f'最优w：\n{w}\n')
    print(f'\n最优b：\n{b}\n\n')

    # 预测并计算在训练集上的正确率
    y_hat = predict(w, b, x_data)
    print('训练集上的分类正确率：{:.2%}'.format(np.mean(y_hat == y_data)))


if __name__ == "__main__":
    main()
