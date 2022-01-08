# -*- coding: utf-8 -*-
"""
ml_100k数据集实用程序

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""

import numpy as np
import distance


def load_dataset(file_name, delimiter=','):
    """
    加载CSV或TSV数据集
    输入参数
        file_name：CSV文件名，delimiter：分隔符
    输出参数
        data：数据集
    """
    with open(file_name) as file:
        content = file.readlines()
        data = np.mat([list(line.strip("\n").split(delimiter)) for line in content])
    return data


def get_uinfo():
    """
    从u.info文件中获取用户数、物品数和评分总数
    """
    file = '../data/ml-100k/u.info'
    with open(file, encoding="utf-8") as fr:
        content = fr.readlines()
        x = [f.split(" ")[: -1] for f in content]
    return int(x[0][0]), int(x[1][0]), int(x[2][0])


def get_films():
    """
    获取全部电影id和名称
    """
    # 获取电影数据结构
    file = '../data/ml-100k/u.item'
    with open(file, encoding="ISO-8859-1") as fr:
        content = fr.readlines()
        x = [f.split("|")[: -1] for f in content]
    return np.array(x)[:, :2]


def get_rating_matrix(file):
    """
    构建评分矩阵
    输入
        file：文件名。如'../data/ml-100k/u1.base'
    输出
        y：给定用户评分矩阵
        r：是否评分矩阵
    """
    # 获取用户数和电影数
    users, items, _ = get_uinfo()

    # 读取训练集。每行格式：用户id、电影id、评分、时间戳
    ratings = load_dataset(file, delimiter='\t')
    # 丢弃时间戳
    ratings = np.array(ratings[:, :3], dtype=int)

    # 构建 用户数 x 电影数 的评分矩阵
    y = np.zeros((users, items), dtype=int)
    r = np.zeros((users, items), dtype=int)

    # 评分数
    n = len(ratings)

    # 循环填写评分
    for i in range(n):
        # 由于用户id和电影id都是从1开始，而Python习惯下标从0开始，因此都减一
        y[ratings[i, 0] - 1, ratings[i, 1] - 1] = ratings[i, 2]
        r[ratings[i, 0] - 1, ratings[i, 1] - 1] = 1

    return y, r


def get_ratings_by_user_id(user_id, file):
    """
    获取给定用户id的电影评分行向量
    输入
        user_id：用户id
        file：文件名
    输出
        y：给定用户评分行向量
        r：是否评分的行向量
    """
    # 读取训练集。每行格式：用户id、电影id、评分、时间戳
    ratings = load_dataset(file, delimiter='\t')
    # 丢弃时间戳
    ratings = np.array(ratings[:, :3], dtype=int)

    # 查找给定用户id对电影的所有评分，用户id，电影id，评分的矩阵
    idx = np.where(ratings[:, 0] == user_id)
    ratings = ratings[idx]

    # 读取u.info文件的电影数
    _, items, _ = get_uinfo()
    y = np.zeros((1, items))
    r = np.zeros((1, items))

    # 循环为y和r赋值
    for i in range(len(ratings)):
        # 由于用户id和电影id都是从1开始，而Python习惯下标从0开始，因此都减一
        one_rating = ratings[i]
        y[0, one_rating[1] - 1] = one_rating[2]
        r[0, one_rating[1] - 1] = 1
    return y, r


def get_k_similar_users(k, y_base, r_base, y_test, r_test):
    """
    获取前K名相似用户
    输入
        k：近邻个数
        y_base, r_base, y_test, r_test
    输出
        similar_users：k个近邻用户
    """
    # 获取皮尔森相似度
    similarity = distance.distance2(y_base, r_base, y_test, r_test, 'pearson')
    s = np.argsort(-similarity[:, 0])
    similar_users = s[: k]
    return similar_users + 1    # 0起始和1起始的差


def get_recommends(y_base, r_base, similar_users):
    """
    由近邻用户id计算推荐的影片
    输入
        y_base：评分矩阵
        r_base：是否评分的标志矩阵
        similar_users：与目标用户相似的用户id
    输出
        films：推荐的影片
    """
    k = len(similar_users)
    y = y_base[similar_users - 1]
    r = r_base[similar_users - 1]
    films = np.zeros(k)
    films = films.astype(np.str)
    # 影片名称列表
    film_names = get_films()
    for i in range(k):
        # 此处不能只使用评过分的rating，因为要根据index得到影片名称
        f = np.argsort(-y[i])
        # 这里使用的策略是近邻用户每人推荐一部影片
        films[i] = film_names[f[0] + 1, 1]
    # 去重
    return set(films)
