# -*- coding: utf-8 -*-
"""
绘制树模型

The Elements of Machine Learning ---- Principles Algorithms and Practices(Python Edition)
@author: Mike Yuan, Copyright 2019~2020
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止plt汉字乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

decisionNode = dict(boxstyle="round4", fc="0.8")
leafNode = dict(boxstyle="square", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(my_tree):
    """
    计算决策树的叶子节点数
    """
    num_leafs = 0

    nodes = list(my_tree.keys())
    first_str = nodes[0]
    second_dict = my_tree[first_str]

    for key in second_dict.keys():
        # 子树分支
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        # 叶子分支
        else:
            num_leafs += 1

    return num_leafs


def get_tree_depth(my_tree):
    """
    计算决策树的深度
    """
    max_depth = 0
    # 节点信息
    nodes = list(my_tree.keys())
    first_str = nodes[0]
    second_dict = my_tree[first_str]

    for key in second_dict.keys():
        # 子树分支
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        # 叶子分支
        else:
            this_depth = 1

        # 更新最大深度
        if this_depth > max_depth:
            max_depth = this_depth

    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    绘制节点和连线
    输入：
        node_txt：终点节点文本，center_pt：终点坐标，parent_pt：起点坐标，node_type: 终点节点类型
    输出：
        无
    """
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                            textcoords='axes fraction', va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    在指定位置显示文本
    输入：
        cntr_pt：终点坐标，parent_pt：起点坐标，txt_string：待显示文本
    输出：
        无
    """
    # 计算中间位置坐标
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]

    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parent_pt, node_txt):
    """
    绘制决策树
    输入：
        my_tree：决策树，parent_pt：起点坐标，node_txt：节点文本
    输出：
        无
    """

    # 叶子节点数
    num_leafs = get_num_leafs(my_tree)

    nodes = list(my_tree.keys())
    first_str = nodes[0]

    # 第一棵子树位置
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)

    # 绘制文本和节点
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decisionNode)

    # 子树
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD

    for key in second_dict.keys():
        # 子树分支
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        # 叶子分支
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leafNode)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))

    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(in_tree):
    """
    显示决策树
    输入：
        in_tree: 决策树字典描述
    输出：
        无
    """

    # 创建新的图像并清空 - 无横纵坐标
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **ax_props)

    # 树的总宽度和总高度
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))

    # 当前绘制节点的坐标
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0

    # 绘制决策树
    plot_tree(in_tree, (0.5, 1.0), '')

    plt.show()
