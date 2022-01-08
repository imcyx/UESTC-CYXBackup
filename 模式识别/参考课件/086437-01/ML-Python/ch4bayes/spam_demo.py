# -*- coding: utf-8 -*-
"""
使用朴素贝叶斯求解垃圾邮件分类问题
伯努利模型

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def load_dataset(file_name, delimiter=','):
    """
    加载CSV或TSV数据集
    输入参数
        file_name：CSV文件名，delimiter：分隔符
    输出参数
        data：数据集
    """
    fr = open(file_name).readlines()
    data = np.array([list(line.strip("\n").split(delimiter)) for line in fr])
    return data.astype(float)


def train_nb(train_x, train_y):
    """
    训练文档分类的朴素贝叶斯模型
    输入参数
        train_x：文档数据集，train_y：文档标签
    输出参数
        p0：类别为0时各个单词的条件概率，p1：类别为1时各个单词的条件概率，phi1：类别为1的概率
    """
    num_doc, word_num = train_x.shape
    phi1 = float(np.sum(train_y)) / float(num_doc)
    # 初值为1，拉普拉斯平滑
    p0_num = np.ones(word_num)
    p1_num = np.ones(word_num)
    p0_denom = 2
    p1_denom = 2

    for doc_idx in range(num_doc):
        x = train_x[doc_idx].copy()
        x = (x > 0).astype(int)
        if train_y[doc_idx] == 1:
            # 正例
            p1_num += x
            p1_denom += np.sum(x)
        else:
            # 负例
            p0_num += x
            p0_denom += np.sum(x)
    p0 = np.log(p0_num / float(p0_denom))
    p1 = np.log(p1_num / float(p1_denom))
    return p0, p1, phi1


def classify_nb(p0, p1, phi1, test_x):
    """
    对测试数据集进行分类
    输入参数
        p0：类别为0时各个单词的条件概率，p1：类别为1时各个单词的条件概率，phi1：类别为1的概率，test_x：测试集
    输出参数
        y_hat：预测输出
    """
    y_hat = []
    for idx in range(len(test_x)):
        x = (test_x[idx].copy() > 0).astype(int)
        prob1 = np.sum(x * p1) + np.log(phi1)
        prob0 = np.sum(x * p0) + np.log(1 - phi1)
        y_hat.append((prob1 > prob0).astype(int))
    return y_hat


def main():
    # 加载数据
    data = load_dataset("../data/spambase/spambase.data")
    n, d = data.shape
    np.random.seed(1234)

    # 划分训练集测试集
    # 随机置乱
    np.random.shuffle(data)
    train_size = round(0.7 * n)
    train_x = data[0: train_size, 0: d - 1]
    train_y = data[0: train_size, d - 1]
    test_x = data[train_size:, 0: d - 1]
    test_y = data[train_size:, d - 1]

    # 训练模型
    p0, p1, phi1 = train_nb(train_x, train_y)

    # 测试
    y_hat = classify_nb(p0, p1, phi1, test_x)
    print('垃圾邮件问题结果：')
    print('测试集上的分类正确率： {:.2%}'.format(np.mean(y_hat == test_y)))


if __name__ == "__main__":
    main()
