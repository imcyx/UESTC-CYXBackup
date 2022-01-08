# -*- coding: utf-8 -*-
"""
使用朴素贝叶斯求解天气问题
都是标称属性

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
    data = np.mat([list(line.strip("\n").split(delimiter)) for line in fr])
    return data


def get_discrete_dataset(data_set, feat_idx, value):
    """
    统计给定离散特征取值的数据集
    输入参数
        data_set：数据集，feat_idx：给定的特征序号, value：值
    输出参数
        sub_data：数据集子集
    """
    sub_data = data_set[np.nonzero(data_set[:, feat_idx] == value)[0], :]
    return sub_data


def is_discrete_feature(data_set, feat_idx):
    """
    测试是否为离散属性
    输入参数
        data_set：数据集，feat_idx：给定的特征序号
    输出参数
        is_str_feat：是否为离散属性
    """
    is_str_feat = False
    try:
        _ = data_set[:, feat_idx].astype(float)
    except ValueError:
        is_str_feat = True

    return is_str_feat


def train(data_set, feature_list, use_smoothing=False):
    """
    训练朴素贝叶斯模型
    输入参数
        data_set：数据集，feature_list：特征名称列表, use_smoothing：使用拉普拉斯平滑
    输出参数
        prob：各个特征的条件概率模型，feat_value：各个特征的取值
    """
    n, d = data_set.shape
    feat_value = {}  # 每个特征的取值
    prob = []  # 条件概率模型
    for idx in range(d):
        if is_discrete_feature(data_set, idx):
            # 只有离散属性，才记录不重复的特征取值
            feat_value[feature_list[idx]] = sorted(set(data_set[:, idx].T.tolist()[0]))

    # 计算类别属性概率
    num_class = len(feat_value.get(feature_list[-1]))
    c_class = np.zeros(num_class)
    for c in range(num_class):
        c_class[c] = len(get_discrete_dataset(data_set, -1, feat_value.get(feature_list[-1])[c]))
    if use_smoothing:
        c_class += 1
    p_class = c_class / (n if not use_smoothing else (n + num_class))

    # 计算条件概率
    # 离散属性计算条件概率，连续属性计算均值方差
    for idx in range(d - 1):
        if is_discrete_feature(data_set, idx):
            # 离散属性
            num_feat_value = len(feat_value.get(feature_list[idx]))
            p = np.zeros((num_class, num_feat_value))
            for c in range(num_class):
                sub_data = get_discrete_dataset(data_set, -1, feat_value.get(feature_list[-1])[c])
                for v in range(len(feat_value.get(feature_list[idx]))):
                    p[c][v] = float((1 if use_smoothing else 0) +
                                    len(get_discrete_dataset(sub_data, idx, feat_value.get(
                                        feature_list[idx])[v]))) / (
                                          len(sub_data) + (num_feat_value if use_smoothing else 0))
            prob.append(p)
        else:
            # 连续属性
            p = np.zeros((num_class, 2))     # 2是因为只需要记录均值和方差
            for c in range(num_class):
                sub_data = get_discrete_dataset(data_set, -1, feat_value.get(feature_list[-1])[c])
                float_feat = sub_data[:, idx].astype(float)
                p[c, 0] = np.mean(float_feat)
                p[c, 1] = np.std(float_feat, ddof=1)
            prob.append(p)

    # 最后加上类别的概率
    prob.append(p_class)
    return prob, feat_value


def classify(prob, feat_value, feature_list, test_set):
    """
    对测试数据集进行分类
    输入参数
        prob：各个特征的条件概率模型，feat_value：各个特征的取值，feature_list：特征名称列表, test_set：测试集
    输出参数
        y_hat：预测输出
    """
    num_class = len(prob[-1])
    class_value = feat_value.get(feature_list[-1])
    n, d = test_set.shape
    y_hat = []
    for i in range(n):
        likelihood = prob[-1].copy()
        for j in range(d):
            if is_discrete_feature(test_set, j):
                # 离散属性
                this_feat_value = feat_value.get(feature_list[j])
                feat_idx = this_feat_value.index(test_set[i, j])
                for c in range(num_class):
                    likelihood[c] *= prob[j][c][feat_idx]
            else:
                # 连续属性
                x = test_set[i, j].astype(float)
                for c in range(num_class):
                    mu = prob[j][c][0]
                    sigma = prob[j][c][1]
                    likelihood[c] *= 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.square(x - mu) / (2 * sigma * sigma))

        y_hat.append(class_value[np.argmax(likelihood)])

    return y_hat


def main():
    # 加载数据
    data = load_dataset("../data/weather_numeric.csv")
    feature_list = ['outlook', 'temperature', 'humidity', 'windy', 'play']
    # 测试数据
    # 书上的例子
    test = np.mat([['sunny', '66', '90', 'TRUE', 'no']])
    test_x = test[:, 0: -1]
    test_y = test[:, -1]
    # 训练模型
    model, feat_value = train(data, feature_list)

    y_hat = classify(model, feat_value, feature_list, test_x)
    print(y_hat)
    print(y_hat == test_y)

    # 使用训练集测试
    test_x = data[:, 0: -1]
    test_y = data[:, -1].squeeze()
    y_hat = classify(model, feat_value, feature_list, test_x)
    print('天气问题结果：')
    print('训练集上的分类正确率： {:.2%}'.format(np.mean(y_hat == test_y)))


if __name__ == "__main__":
    main()
