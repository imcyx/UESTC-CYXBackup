# Common imports
import os
import numpy as np
import warnings
from scipy import sparse

# To plot pretty figures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Make dir and config save
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# data preprocess
def preprocess(data_name):
    x, y = [], []
    with open(data_name, "r") as test_datasets:
        for data in test_datasets.readlines():
            data = data.strip("\n")
            if data and data[0].isnumeric():
                data = data.split(",")
                data_x = [int(data_) for data_ in data]
            else:
                continue
            x.append(data_x)
    x = np.array(x)
    return x


class KMeans(object):
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = np.random.RandomState(random_state)
        self.mean_vector = None

    def row_norms(self, X, squared=False):
        '''计算行向量点积

        :param X: Input Array
        :param squared: whether square result
        :return: norms
        '''
        # aij * bij -> ci
        # Here a is same as b, so the result is ci=sum(ai^2)
        norms = np.einsum('ij,ij->i', X, X)

        # If squared, Vector result means each axis distance
        if not squared:
            np.sqrt(norms, norms)
        return norms

    def safe_sparse_dot(self, a, b, *, dense_output=False):
        """正确处理稀疏矩阵情况的点积。

        Parameters
        ----------
        a : {ndarray, sparse matrix}
        b : {ndarray, sparse matrix}
        dense_output : bool, default=False
            When False, ``a`` and ``b`` both being sparse will yield sparse output.
            When True, output will always be a dense array.

        Returns
        -------
        dot_product : {ndarray, sparse matrix}
            Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
        """
        if a.ndim > 2 or b.ndim > 2:
            if sparse.issparse(a):
                # sparse is always 2D. Implies b is 3D+
                # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
                b_ = np.rollaxis(b, -2)
                b_2d = b_.reshape((b.shape[-2], -1))
                ret = a @ b_2d
                ret = ret.reshape(a.shape[0], *b_.shape[1:])
            elif sparse.issparse(b):
                # sparse is always 2D. Implies a is 3D+
                # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
                a_2d = a.reshape(-1, a.shape[-1])
                ret = a_2d @ b
                ret = ret.reshape(*a.shape[:-1], b.shape[1])
            else:
                ret = np.dot(a, b)
        else:
            ret = a @ b

        if (sparse.issparse(a) and sparse.issparse(b)
                and dense_output and hasattr(ret, "toarray")):
            return ret.toarray()
        return ret

    def euclidean_distances(self, X, Y=None):
        """
            将 X（和 Y=X）的行视为向量，计算每对向量之间的距离矩阵。
            出于效率原因，一对行向量 x 和 y 之间的欧几里德距离计算如下：
                dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

            与其他计算距离的方法相比，此公式有两个优点。
            首先，它在处理稀疏数据时计算效率高。
            其次，如果一个参数发生变化而另一个参数保持不变，则可以预先计算“dot(x, x)” 或者 “dot(y, y)”。
        """
        array = X
        array_orig = X
        if Y is X or Y is None:
            Y = np.array(array, dtype=array_orig.dtype)

        XX = self.row_norms(X, squared=True)[:, np.newaxis]
        YY = self.row_norms(Y, squared=True)[np.newaxis, :]

        distances = - 2 * self.safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
        distances = distances.astype(np.float64)
        np.maximum(distances, 0, out=distances)

        # 确保向量和自身之间的距离设置为0.0。
        # 由于浮点舍入错误，可能不是这样。
        if X is Y:
            np.fill_diagonal(distances, 0)

        # 返回平方根的数值
        return np.sqrt(distances, out=distances)

    def stable_cumsum(self, arr, axis=None, rtol=1e-05, atol=1e-08):
        """使用高精度求和向量中给定轴元素，并检查最终值是否与sum匹配。
        Use high precision for cumsum and check that final value matches sum.

        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat.
        axis : int, default=None
            Axis along which the cumulative sum is computed.
            The default (None) is to compute the cumsum over the flattened array.
        rtol : float, default=1e-05
            Relative tolerance, see ``np.allclose``.
        atol : float, default=1e-08
            Absolute tolerance, see ``np.allclose``.
        """
        out = np.cumsum(arr, axis=axis, dtype=np.float64)
        expected = np.sum(arr, axis=axis, dtype=np.float64)
        if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                                 atol=atol, equal_nan=True)):
            warnings.warn('cumsum was found to be unstable: '
                          'its last element does not correspond to sum',
                          RuntimeWarning)
        return out

    def kmeans_plusplus(self, X):
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=X.dtype)
        # 设置本地种子试验的次数
        n_local_trials = 2 + int(np.log(self.n_clusters))

        # 随机选取第一个中心并跟踪点的索引
        center_id = self.random_state.randint(n_samples)
        indices = np.full(self.n_clusters, -1, dtype=int)
        # 生成第一个索引对应的第一个点
        centers[0] = X[center_id]
        indices[0] = center_id

        # 初始化最近距离列表并计算当前潜力
        # 计算随机生成中心点与所有点的距离
        closest_dist_sq = self.euclidean_distances(centers[0, np.newaxis], X)
        # 求出与所有点的距离和
        current_pot = closest_dist_sq.sum()

        # 选择剩余的 n_clusters-1 个点
        for c in range(1, self.n_clusters):
            # 通过抽样选择候选中心，抽样概率与距离最近的现有中心的平方距离成正比
            # 随机产生若干抽样个与前一个中心点距离和成比例的距离值
            rand_vals = self.random_state.random_sample(n_local_trials) * current_pot
            # 找到与这些随机距离值最接近的样本点索引，这些点即被确定为下一次候选中心
            candidate_ids = np.searchsorted(self.stable_cumsum(closest_dist_sq),
                                            rand_vals)

            # 数值不精确可能导致候选id超出范围，不能超过最大索引距离
            np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                    out=candidate_ids)

            # 计算所有点到每个候选中心的距离
            distance_to_candidates = self.euclidean_distances(X[candidate_ids], X)

            # 更新每个候选中心的最近距离平方和矩阵
            # 每个样本点到min(候选点, 上一个中心点)的距离
            np.minimum(closest_dist_sq, distance_to_candidates,
                       out=distance_to_candidates)
            # 距离和矩阵
            candidates_pot = distance_to_candidates.sum(axis=1)

            # 通过最近距离平方和决定哪个候选点是最好的
            best_candidate = np.argmin(candidates_pot)
            # 最近的距离和
            current_pot = candidates_pot[best_candidate]
            # 最近的距离矩阵
            closest_dist_sq = distance_to_candidates[best_candidate]
            # 最近的点
            best_candidate = candidate_ids[best_candidate]

            # 添加在尝试中找到的最佳中心候选点
            if sparse.issparse(X):
                centers[c] = X[best_candidate].toarray()
            else:
                centers[c] = X[best_candidate]
            indices[c] = best_candidate

        return centers, indices

    def kmeans_divide_cluster(self, X, centers):
        '''
        根据计算的均值向量重新划分簇
        :param X: Input sample array
        :param centers: Center Vectors loc
        :return:
        '''
        # 计算所有样本点与均值向量的距离
        distance_to_candidates = self.euclidean_distances(X, centers)
        # 计算距离最近的均值向量确定簇标记
        index_list = np.argmin(distance_to_candidates, axis=1)
        # 生成n个簇数组
        divide_list = [[] for s in range(len(centers))]
        # 划分簇
        for i, s in enumerate(X):
            index = divide_list[index_list[i]]
            index.append(s.tolist())

        return divide_list

    def kmeans_one_from_cluster(self, dot):
        '''
        计算所属簇
        :param dot: one dot
        :return:
        '''
        # 计算所有样本点与均值向量的距离
        distance_to_candidates = self.euclidean_distances(dot, self.mean_vector)
        # 计算距离最近的均值向量确定簇标记
        index_list = np.argmin(distance_to_candidates, axis=1)
        return index_list

    def kmeans_update_MeanVector(self, center_list, divide_list):
        '''
        更新均值向量
        :param center_list: Center Vectors loc
        :param divide_list: Dots belonging lists
        :return:
        '''
        raw_mean_vector = center_list
        self.mean_vector = np.empty([len(divide_list), len(divide_list[0][0])])
        # 更新均值向量
        for i, s in enumerate(divide_list):
            cluster = np.array(s)
            self.mean_vector[i] = cluster.mean(axis=0)
        # 判断是否还需要更新
        if not np.all(raw_mean_vector - self.mean_vector):
            return -1, self.mean_vector
        else:
            return 0, self.mean_vector


def plot_data(X, color='k', subplt=plt):
    '''
    绘制样本点集
    :param X: Input sample dots
    :param color: color plot
    :param subplt: which subplot plot
    :return:
    '''
    subplt.plot(X[:, 0], X[:, 1], f'{color}x', markersize=2)

def plot_centroids(centroids, circle_color='w', cross_color='r', subplt=plt):
    '''
    绘制中心点
    :param centroids: Center Vectors loc
    :param circle_color: circle_color
    :param cross_color: cross_color
    :param subplt: which subplot plot
    :return:
    '''
    subplt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    subplt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(kmeans, centers, X, divide_list,
                             resolution=1000, show_centroids=True, subplt=plt):
    '''
    绘制样本区域分界
    :param kmeans: kmeans Category，ask classification result
    :param centers: Center Vectors loc
    :param X: Input sample dots
    :param divide_list: Dots belonging lists
    :param resolution: step for sampling boundaries
    :param show_centroids: whether draw centroies
    :param subplt: which subplot plot
    :return:
    '''
    # 获取颜色表
    color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    color_label = list(color_dict.keys())

    # 生成取样点
    mins = X.min(axis=0) - 0.5
    maxs = X.max(axis=0) + 0.5
    # 转换成坐标
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    dots = np.c_[xx.ravel(), yy.ravel()]
    # 预测取样点所属分类矩阵
    Z = kmeans.kmeans_one_from_cluster(dots)
    # 将1维分类矩阵转换成2维
    # 因为送入的xx，yy采样点是二维的，所以将处理完的结果重新转换回二维之后，
    # 每个元素对应的即为（xx, yy）
    Z = Z.reshape(xx.shape)

    # 绘制分界背景
    subplt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),cmap="Pastel2")
    # 绘制分界线
    subplt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),linewidths=1, colors='k')

    # 绘制样本点
    for i, s in enumerate(divide_list):
        plot_data(np.array(s), color=color_label[i%8], subplt=subplt)
    # 绘制均值向量
    if show_centroids:
        plot_centroids(centers, subplt=subplt)

    # if show_xlabels:
    #     subplt.xlabel("$x_1$", fontsize=14)
    # else:
    #     subplt.tick_params(labelbottom=False)
    # if show_ylabels:
    #     subplt.ylabel("$x_2$", fontsize=14, rotation=0)
    # else:
    #     subplt.tick_params(labelleft=False)

X= preprocess('K-means.data')

kmeans = KMeans(n_clusters=5, random_state=42)
centers, _ = kmeans.kmeans_plusplus(X)
i=0
while(True):
    i+=1
    divide_list = kmeans.kmeans_divide_cluster(X, centers)
    res, centers = kmeans.kmeans_update_MeanVector(centers, divide_list)
    if res < 0:
        break

fig = plt.figure(figsize=(14, 7))
# plt.xticks(np.linspace(0, 20, 21))
# plt.yticks(np.linspace(0, 20, 21))

plt1 = fig.add_subplot(211)
plt1.scatter(X[:,0], X[:, 1], s=6, marker='.')
plt1.set_xlabel("$x_1$", fontsize=14)
plt1.set_ylabel("$x_2$", fontsize=14, rotation=0)
plt1.set_xlim(-0.5, 20.5)

plt2 = fig.add_subplot(212)
plot_decision_boundaries(kmeans, centers, X, divide_list, subplt=plt2)
plt2.set_xlabel("$x_1$", fontsize=14)
plt2.set_ylabel("$x_2$", fontsize=14, rotation=0)
# plot_decision_boundaries(kmeans, centers, X, divide_list)
# plt.xlabel("$x_1$", fontsize=14)
# plt.ylabel("$x_2$", fontsize=14, rotation=0)

save_fig("k-means")
plt.show()



