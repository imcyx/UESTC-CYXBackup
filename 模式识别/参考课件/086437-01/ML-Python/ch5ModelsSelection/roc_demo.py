# -*- coding: utf-8 -*-
"""
ROC分析

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import lr_function
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def main():
    # 返回随机数据集的真实标签和预测标签
    y_real, y_pred = lr_function.lr_classifier()
    print('交叉验证的分类正确率：: {:.2%}'.format(np.mean((y_pred >= 0.5) == y_real)))

    # 运行ROC分析
    # 阈值
    threshold = np.linspace(min(y_pred), max(y_pred), 1000)
    tpr = np.zeros(len(threshold))
    fpr = np.zeros(len(threshold))
    for i in range(len(threshold)):
        binary_pred = y_pred >= threshold[i]
        # 计算真阳性、假阳性、真阴性、假阴性
        tp = np.sum((binary_pred == 1) & (y_real == 1))
        fp = np.sum((binary_pred == 1) & (y_real == 0))
        tn = np.sum((binary_pred == 0) & (y_real == 0))
        fn = np.sum((binary_pred == 0) & (y_real == 1))
        # 计算TPR和FPR
        tpr[i] = tp / (tp + fn)
        fpr[i] = fp / (tn + fp)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, 'r')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('FPR(%)')
    plt.ylabel('TPR(%)')
    plt.show()

    # 计算AUC
    auc = np.sum(-0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]))
    print(f'\nAUC: {auc}')


if __name__ == "__main__":
    main()
