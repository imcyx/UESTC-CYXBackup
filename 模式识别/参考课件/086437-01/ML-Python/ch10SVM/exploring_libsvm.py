# -*- coding: utf-8 -*-
"""
探索LibSVM

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import sys

sys.path.append(r'D:\libsvm-3.24\python')
sys.path.append(r'D:\libsvm-3.24\tools')

import svmutil
import grid


def main():
    # 1、直接训练和测试
    y, x = svmutil.svm_read_problem('../data/libsvm_guide/train.1')
    yt, xt = svmutil.svm_read_problem('../data/libsvm_guide/test.1')
    m = svmutil.svm_train(y, x)
    svmutil.svm_predict(yt, xt, m)
    # 结果：Accuracy = 66.925% (2677/4000) (classification)

    # 2、命令行运行svm-scale
    # svm-scale -l -1 -u 1 -s range1 train.1 > train.1.scale
    # svm-scale -r range1 test.1 > test.1.scale

    # 3、使用缩放后的数据训练和测试
    y, x = svmutil.svm_read_problem('../data/libsvm_guide/train.1.scale')
    yt, xt = svmutil.svm_read_problem('../data/libsvm_guide/test.1.scale')
    m = svmutil.svm_train(y, x)
    svmutil.svm_predict(yt, xt, m)
    # 结果：Accuracy = 96.15% (3846/4000) (classification)

    # 4、寻找最优参数
    rate, param = grid.find_parameters('../data/libsvm_guide/train.1.scale', '-log2c -3,3,1 -log2g -3,3,1')

    # 5、按照最优参数再次训练和测试
    y, x = svmutil.svm_read_problem('../data/libsvm_guide/train.1.scale')
    yt, xt = svmutil.svm_read_problem('../data/libsvm_guide/test.1.scale')
    m = svmutil.svm_train(y, x, f'-c {param.get("c")} -g {param.get("g")}')
    svmutil.svm_predict(yt, xt, m)
    # 结果：Accuracy = 97.15% (3886/4000) (classification)

    # 6、自己编程实现寻找最优参数
    # 寻优参数C和gamma
    best_acc = 0
    for log2c in range(-3, 6):
        for log2g in range(-3, 6):
            options = f'-q -t 2 -c {2. ** log2c} -g {2. ** log2g}'
            model = svmutil.svm_train(y, x, options)
            _, acc, _ = svmutil.svm_predict(yt, xt, model)
            if acc[0] > best_acc:
                best_acc = acc[0]
                best_c = 2. ** log2c
                best_g = 2. ** log2g
            print(f'当前 {log2c} {log2g} {acc[0]} (最优 c={best_c}, g={best_g}, 准确率={best_acc}%)\n')


if __name__ == "__main__":
    main()
