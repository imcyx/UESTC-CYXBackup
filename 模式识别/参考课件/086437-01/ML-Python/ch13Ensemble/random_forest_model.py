# -*- coding: utf-8 -*-
"""
随机森林模型

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
from decision_tree_model import ClassificationTree


class RandomForest:
    """ 随机森林分类器
    参数：
    n_tree：分类树的数量
    max_features：每棵分类树允许的最大的特征数
    min_samples：训练决策树时分裂所需的最小样本数
    min_gain：树进一步分裂所要求的最小增益（不纯度）阈值
    max_depth：每棵树的最大深度
    """

    def __init__(self, n_tree=100, min_samples=2, min_gain=0,
                 max_depth=float("inf"), max_features=None):
        self.n_tree = n_tree
        self.min_samples = min_samples
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features

        self.trees = []
        # 迭代构建森林的每棵树
        for _ in range(self.n_tree):
            tree = ClassificationTree(min_samples=self.min_samples,
                                      min_impurity=self.min_gain, max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, x, y):
        """ 使用自助抽样数据集和随机特征训练随机森林 """
        sub_sets = self.get_bootstrap_data(x, y)
        n_features = len(x[0])
        # 计算最大的特征数
        if self.max_features is None:
            # 如果不指定，则默认为总特征数的平方根
            self.max_features = int(np.sqrt(n_features))
        # 迭代训练每一棵树
        for i in range(self.n_tree):
            # 随机抽样选取特征
            sub_x, sub_y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=True)
            sub_x = sub_x[:, idx]
            self.trees[i].fit(sub_x, sub_y)
            self.trees[i].feature_idx = idx
            # print(f"第{i}棵树训练完毕！")

    def predict(self, x):
        """ 使用训练好的随机森林模型进行预测 """
        y_preds = []        # 暂存全部决策树的预测结果
        for i in range(self.n_tree):
            idx = self.trees[i].feature_idx
            sub_x = x[:, idx]
            y_hat = self.trees[i].predict(sub_x)
            y_preds.append(y_hat)
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            # bincount函数统计每个索引出现的次数
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return y_pred

    def get_bootstrap_data(self, x, y):
        """ 自助抽样得到n_tree组数据集 """
        n = len(x)
        results_data_sets = []
        for _ in range(self.n_tree):
            idx = np.random.choice(n, n, replace=True)
            bs_x = x[idx]
            bs_y = y[idx]
            results_data_sets.append([bs_x, bs_y])
        return results_data_sets
