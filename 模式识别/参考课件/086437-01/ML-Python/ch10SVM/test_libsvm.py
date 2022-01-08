# -*- coding: utf-8 -*-
"""
测试LibSVM是否正确安装的小程序

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import sys

sys.path.append(r'D:\libsvm-3.24\python')

import svmutil

# 加载数据
y, x = svmutil.svm_read_problem(r'D:\libsvm-3.24\heart_scale')
# 训练模型
model = svmutil.svm_train(y[:200], x[:200], '-c 4')
# 预测
p_label, p_acc, p_val = svmutil.svm_predict(y[200:], x[200:], model)

