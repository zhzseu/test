"""
K-Means聚类算法学习与实践
------------------------

聚类（Clustering）是机器学习中的无监督学习方法之一，旨在将数据集中的样本划分为若干个相似的组（簇）。
K-Means是一种基于距离的聚类算法，常用于数据挖掘、用户分群、市场分析等领域。

K-Means算法的基本步骤：
1. 随机选择K个样本作为初始簇中心（质心）。
2. 将每个样本分配到距离其最近的簇中心。
3. 根据分配结果，更新每个簇的质心。
4. 重复步骤2和3，直到质心收敛或达到最大迭代次数。

数学原理：
- 目标函数：最小化所有样本到其所属簇中心的距离平方和。
- 公式：J = sum_{i=1}^K sum_{x in C_i} ||x - mu_i||^2

优缺点：
- 优点：实现简单、收敛快、适用于大数据。
- 缺点：对初始值敏感、难以处理不均衡数据、需预先指定K值、只适用于凸形簇。

本示例包括：
1. 手写K-Means算法实现（不依赖scikit-learn）
2. 用鸢尾花数据集演示手写算法效果
3. 用scikit-learn的KMeans类实现聚类并对比效果
4. 可视化聚类结果
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# ========================
# 1. 距离计算函数
# ========================
def distance(u, v, p=2):
    """
    计算两个向量的距离
    默认使用欧氏距离（p=2），也可设置为曼哈顿距离等
    """
    return np.sum(np.abs(u - v) ** p) ** (1 / p)

# ========================
# 2. 初始化质心
# ========================
def init_centroids(X, k):
    """
    随机选择k个样本作为初始质心
    """
    index = np.random.choice(np.arange(len(X)), k, replace=False)
    return X[index]

# ========================
# 3. 找到样本最近的质心
# ========================
def closest_centroid(sample, centroids):
    """
    对于一个样本，返回它最近的质心索引
    """
    distances = [distance(sample, centroid) for centroid in centroids]
    return np.argmin(distances)

# ========================
# 4. 根据质心分组
# ========================
def build_clusters(X, centroids):
    """
    将所有样本分配到最近的质心，形成簇
    返回每个簇内的样本索引
    """
    clusters = [[] for _ in range(len(centroids))]
    for idx, sample in enumerate(X):
        centroid_index = closest_centroid(sample, centroids)
        clusters[centroid_index].append(idx)
    return clusters

# ========================
# 5. 更新质心
# ========================
def update_centroids(X, clusters):
    """
    根据每个簇的成员，计算新的质心
    """
    return np.array([np.mean(X[cluster], axis=0) for cluster in clusters])

# ========================
# 6. 生成标签
# ========================
def make_label(X, clusters):
    """
    为每个样本生成标签（属于哪个簇）
    """
    labels = np.zeros(len(X))
    for idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            labels[sample_idx] = idx
    return labels

# ========================
# 7. KMeans主函数
# ========================
def kmeans(X, *, k, max_iter=1000, tol=1e-4):
    """
    KMeans聚类算法
    参数:
        X: 输入数据，shape=(n_samples, n_features)
        k: 簇的数量
        max_iter: 最大迭代次数
        tol: 收敛容忍度
    返回:
        labels: 每个样本所属的簇标签
        centroids: 最终各簇的质心
    """
    centroids = init_centroids(X, k)
    for _ in range(max_iter):
        clusters = build_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters)
        # 若质心变化很小则提前结束
        if np.allclose(new_centroids, centroids, rtol=tol):
            break
        centroids = new_centroids
    labels = make_label(X, clusters)
    return labels, centroids

# ========================
# 8. 用鸢尾花数据集测试自实现KMeans
# ========================
iris = load_iris()
X, y = iris.data, iris.target

labels, centers = kmeans(X, k=3)

# ========================
# 9. 可视化自实现KMeans结果
# ========================
colors = ['#FF6969', '#050C9C', '#365E32']
markers = ['o', 'x', '^']

plt.figure(dpi=300)
for i in range(len(centers)):
    samples = X[labels == i]
    plt.scatter(samples[:, 2], samples[:, 3], marker=markers[i], color=colors[i], label=f'Cluster {i+1}')
    plt.scatter(centers[i, 2], centers[i, 3], marker='*', color='r', s=120)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Custom KMeans Clustering')
plt.legend()
plt.show()

# ========================
# 10. 可视化原始标签（对比效果）
# ========================
plt.figure(dpi=300)
for i in range(3):
    samples = X[y == i]
    plt.scatter(samples[:, 2], samples[:, 3], marker=markers[i], color=colors[i], label=f'Class {i+1}')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Ground Truth')
plt.legend()
plt.show()

# ========================
# 11. 用scikit-learn库的KMeans对比
# ========================
km_cluster = KMeans(
    n_clusters=3,       # 簇的数量
    max_iter=30,        # 最大迭代次数
    n_init=10,          # 初始化次数
    init='k-means++',   # 质心选择方法
    algorithm='elkan',  # 优化算法
    tol=1e-4,           # 容忍度
    random_state=3      # 随机种子
)

"""
对`KMeans`类的几个超参数加以说明：

1. `n_clusters`：指定聚类的簇数，即 k 值，默认值为`8`。
2. `max_iter`：最大迭代次数，默认值为`300`，控制每次初始化中 K-Means 迭代的最大步数。
3. `init`：初始化质心的方法，默认值为`'k-means++'`，表示从数据中多次随机选取 K 个质心，每次都计算这一次选中的中心点之间的距离，然后取距离最大的一组作为初始化中心点，推荐大家使用这个值；
如果设置为`'random'`则随机选择初始质心。
4. `n_init`：和上面的参数配合，指定算法运行的初始化次数，默认值为`10`。
5. `algorithm`：K-Means 的计算算法，默认值为`'lloyd'`。还有一个可选的值为`'elkan'`，表示基于三角不等式的优化算法，适用于 K 值较大的情况，计算效率较高。
6. `tol`：容忍度，控制算法的收敛精度，默认值为`1e-4`。如果数据集较大时，可适当增大此值以加快收敛速度。

"""

km_cluster.fit(X)
print("Scikit-learn KMeans Labels:\n", km_cluster.labels_)
print("Scikit-learn KMeans Centers:\n", km_cluster.cluster_centers_)
print("Scikit-learn KMeans Inertia（总距离平方和）:\n", km_cluster.inertia_)

# ========================
# 12. 总结
# ========================
"""
K-Means是一种经典的聚类算法，适用于无监督学习任务。
- 主要思想是最小化簇内样本到质心的距离平方和。
- 算法简单、速度快，但对初始质心敏感，对噪声和异常点较为敏感。

实际应用中，建议直接使用sklearn库中的KMeans类，性能更优且参数丰富。
"""
