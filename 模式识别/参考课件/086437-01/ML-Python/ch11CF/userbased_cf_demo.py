# -*- coding: utf-8 -*-
"""
基于用户的协同过滤算法示例

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import ml100k_utils


def main():
    # 训练集
    y_base, r_base = ml100k_utils.get_rating_matrix('../data/ml-100k/u1.base')
    # 测试集。第1个用户的评分
    y_test, r_test = ml100k_utils.get_ratings_by_user_id(1, '../data/ml-100k/u1.test')

    # 计算前K个相似用户
    k = 5
    similar_users = ml100k_utils.get_k_similar_users(k, y_base, r_base, y_test, r_test)

    print(f'前 {k} 个相似用户id：')
    for user in similar_users:
        print(user)

    # 计算推荐
    films = ml100k_utils.get_recommends(y_base, r_base, similar_users)
    print('\n推荐的电影：\n')
    for film in films:
        print(film)


if __name__ == "__main__":
    main()
