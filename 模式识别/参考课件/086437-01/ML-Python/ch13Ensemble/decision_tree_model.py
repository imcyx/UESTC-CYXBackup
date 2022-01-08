# -*- coding: utf-8 -*-
"""
决策树模型

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np


def divide_on_feature(x, feature_idxdx, threshold):
    """ 根据样本的指定特征索引值是否大于给定阈值来划分数据集 """
    def split_func(sample):
        if isinstance(threshold, int) or isinstance(threshold, float):
            return sample[feature_idxdx] >= threshold
        else:
            return sample[feature_idxdx] == threshold

    x_left = np.array([sample for sample in x if split_func(sample)])
    x_right = np.array([sample for sample in x if not split_func(sample)])

    return np.array([x_left, x_right])


def calculate_entropy(y):
    """ 计算标签y的交叉熵 """
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        p = len(y[y == label]) / len(y)
        entropy += -p * np.log2(p)
    return entropy


class DecisionNode:
    """ 决策树中的决策节点或叶节点类
    参数：
    feature_idx：作为阈值度量的特征索引
    threshold：该阈值用于比较指定特征索引的值以确定预测值
    value：预测的类别
    left_branch：左子树，特征值与阈值一致
    right_branch：右子树，特征值与阈值不一致
    """

    def __init__(self, feature_idx=None, threshold=None, value=None,
                 left_branch=None, right_branch=None):
        self.feature_idx = feature_idx      # 要测试的特征索引
        self.threshold = threshold
        self.value = value
        self.left_branch = left_branch      # 左子树
        self.right_branch = right_branch    # 右子树


class DecisionTree(object):
    """ 分类树父类
    参数：
    min_samples：训练决策树时分裂所需的最小样本数
    min_impurity：树进一步分裂所要求的最小不纯度阈值
    max_depth：树的最大深度
    loss：计算不纯度的损失函数
    """

    def __init__(self, min_samples=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # 决策树里的根节点
        self.min_samples = min_samples
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        # 不纯度的计算函数，如信息增益或基尼系数等
        self._impurity_calculation = None
        # 叶节点预测y值的函数，分类树会选取出现最多次数的值
        self._leaf_value_calculation = None
        # y为独热码(multi-dim)，否则(one-dim)
        self.one_dim = None
        # 使用Gradient Boost
        self.loss = loss

    def fit(self, x, y, loss=None):
        """ 训练决策树 """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(x, y)
        self.loss = None

    def _build_tree(self, x, y, current_depth=0):
        """ 构建决策树的递归方法，根据不纯度最好地划分数据 """
        largest_impurity = 0
        best_criteria = None        # 特征索引和阈值
        best_sets = None            # 数据子集

        # 检查是否需要对y增加一维
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # 合并x和y为一个矩阵
        xy = np.concatenate((x, y), axis=1)

        n_samples, n_features = np.shape(x)

        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            # 计算每个特征的不纯度
            for feature_idx in range(n_features):
                # 选定特征列的全部取值
                feature_values = np.expand_dims(x[:, feature_idx], axis=1)
                unique_values = np.unique(feature_values)

                # 遍历选定特征列的每个取值，计算其不纯度
                for threshold in unique_values:
                    # 根据x在索引feature_idx的特征值是否满足阈值来划分x和y
                    xy1, xy2 = divide_on_feature(xy, feature_idx, threshold)

                    if len(xy1) > 0 and len(xy2) > 0:
                        # 选择两个子集的y值
                        y1 = xy1[:, n_features:]
                        y2 = xy2[:, n_features:]

                        # 计算不纯度
                        impurity = self._impurity_calculation(y, y1, y2)

                        # 如果当前信息增益大于以前的最大信息增益，则保存阈值和特征索引
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_idx": feature_idx, "threshold": threshold}
                            best_sets = {
                                "left_x": xy1[:, :n_features],      # 左子树的x
                                "left_y": xy1[:, n_features:],      # 左子树的y
                                "right_x": xy2[:, :n_features],     # 右子树的x
                                "right_y": xy2[:, n_features:]      # 右子树的y
                            }

        if largest_impurity > self.min_impurity:
            # 为左右分支构建子树
            left_branch = self._build_tree(best_sets["left_x"], best_sets["left_y"], current_depth + 1)
            right_branch = self._build_tree(best_sets["right_x"], best_sets["right_y"], current_depth + 1)
            return DecisionNode(feature_idx=best_criteria["feature_idx"], threshold=best_criteria["threshold"],
                left_branch=left_branch, right_branch=right_branch)

        # 如果到达叶节点，就判断其类别
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """ 沿着树往下做递归搜索，根据最后得到的叶节点的值来预测数据样本的类别 """
        if tree is None:
            tree = self.root

        # 如果到达叶节点，就会得到一个值，就返回该值作为预测类别
        if tree.value is not None:
            return tree.value

        # 选择要测试的特征
        feature_value = x[tree.feature_idx]

        # 判断从左分支还是右分支向下搜索
        branch = tree.right_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch

        # 测试子树
        return self.predict_value(x, branch)

    def predict(self, x):
        """ 预测 """
        y_pred = []
        for sample in x:
            y_pred.append(self.predict_value(sample))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ 迭代打印决策树 """
        if not tree:
            tree = self.root

        # 如果到达叶节点，就打印标签
        if tree.value is not None:
            print(tree.value)
        # 继续向树下深入
        else:
            # 打印测试
            print("%s : %s? " % (tree.feature_idx, tree.threshold))
            # 打印左子树
            print("%s左->" % indent, end="")
            self.print_tree(tree.left_branch, indent + indent)
            # 打印右子树
            print("%s右->" % indent, end="")
            self.print_tree(tree.right_branch, indent + indent)


class ClassificationTree(DecisionTree):
    """ 分类树 """
    def _calculate_information_gain(self, y, y1, y2):
        """ 计算信息增益 """
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)
        # print("信息增益：", info_gain)
        return info_gain

    def _majority_vote(self, y):
        """ 多数投票法 """
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # 对样本标签计数
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        # print("最多的投票：", most_common)
        return most_common

    def fit(self, x, y):
        """ 训练分类树 """
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(x, y)


