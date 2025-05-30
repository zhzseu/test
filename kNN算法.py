######### 一、KNN分类算法 ##########

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# 加载鸢尾花数据集
iris = load_iris()
# 查看数据集的介绍


# 特征（150行4列的二维数组，分别是花萼长、花萼宽、花瓣长、花瓣宽）
X = iris.data
# 标签（150个元素的一维数组，包含0、1、2三个值分别代表三种鸢尾花）
y = iris.target
data = np.hstack((X, y.reshape(-1, 1)))
np.random.shuffle(data)

train_size = int(y.size * 0.8)
train, test = data[:train_size], data[train_size:]
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

########### 1.基于numpy实现kNN分类 ##############

def euclidean_distance(u, v):
    return np.sqrt(np.sum(np.abs(u-v)**2))

from scipy import stats

def make_label(X_train, y_train, X_one, k) :
    """
    根据历史数据中k个最近邻为新数据生成标签
    :param X_train: 训练集中的特征
    :param y_train: 训练集中的标签
    :param X_one: 待预测的样本（新数据）特征
    :param k: 邻居的数量
    :return: 为待预测样本生成的标签（邻居标签的众数）
    """
    # 计算x跟每个训练样本的距离
    distes = [euclidean_distance(X_one, X_i) for X_i in X_train]
    # 通过一次划分找到k个最小距离对应的索引并获取到相应的标签
    labels = y_train[np.argpartition(distes, k - 1)[:k]]
    # 获取标签的众数
    return stats.mode(labels).mode

def predict_by_knn(X_train, y_train, X_new, k=5) :
    """
    KNN算法
    :param X_train: 训练集中的特征
    :param y_train: 训练集中的标签
    :param X_new: 待预测的样本构成的数组
    :param k: 邻居的数量（默认值为5）
    :return: 保存预测结果（标签）的数组
    """
    return np.array([make_label(X_train, y_train, X, k) for X in X_new])
#测试
y_pred = predict_by_knn(X_train, y_train, X_test)
print(y_pred == y_test)


########### 2.基于scikit-learn实现 ############

from sklearn.neighbors import KNeighborsClassifier
# 创建模型
model = KNeighborsClassifier()
# 训练模型
model.fit(X_train, y_train)
# 测试
y_pred = model.predict(X_test)

print(y_pred == y_test)

######### 3.模型效果评估 #########
print(y_test)
print(y_pred)

######### a.输出混淆矩阵以及评估报告 ##########
from sklearn.metrics import classification_report, confusion_matrix
print('混淆矩阵')
print(confusion_matrix(y_test, y_pred))
print('评估报告')
print(classification_report(y_test, y_pred))

######### 报告可视化 ##########
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# 创建混淆矩阵显示对象
cm_display_obj = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=iris.target_names)
# 绘制并显示混淆矩阵
cm_display_obj.plot(cmap=plt.cm.Reds)
plt.show()

######### b.二分类问题绘制ROC曲线并显示AUC值 ##########
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay

# 手动构造一组真实值和对应的预测值
y_test_ex = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0])
y_pred_ex = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1])
# 通过roc_curve函数计算出FPR（假正例率）和TPR（真正例率）
fpr, tpr, _ = roc_curve(y_test_ex, y_pred_ex)

# 通过auc函数计算出AUC值并通过RocCurveDisplay类绘制图形
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr)).plot()
plt.show()

########## 4.参数调整 ##########
'''
kNN 算法有两个关键问题，一是距离的度量，二是 k 值的选择。
我们使用 scikit-learn 的KNeighborsClassifier创建分类器模型时，
可以对模型的超参数进行设置，这里有几个比较重要的参数：

n_neighbors：近邻的数量，就是 kNN 算法中 k 的值。
weights：可以选择uniform或distance，前者表示所有样本的权重相同，后者表示距离越近权重越高，默认值是uniform。当然，我们也可以通过传入自定义的函数来确定每个样本的权重。
algorithm：有auto、ball_tree、kd_tree、brute四个选项，默认值为auto。
    其中ball_tree是一种树形结构，基于球体划分的方法将数据点分配到层次化的树结构中，在高维数据和稀疏数据场景下有较好的性能；
    kd_tree也是一种树形结构，通过选择一个维度将空间划分为若干个子区域再进行搜索，从而避免跟所有的邻居进行比较，对于低维度和空间分布均匀的数据，后者有较好的效果，在高维空间中会遇到的维度灾难问题；
    auto选项是根据输入数据的维度自动选择ball_tree或kd_tree；
    brute选项则是使用暴力搜索算法（穷举法），再处理小数据集时，它是一个简单且有效的选择。
leaf_size：使用ball_tree或kd_tree算法时，该参数用于限制树结构叶子节点最大样本数量，默认值为30，该参数会影响树的构建和节点查找的性能。
p：闵可夫斯基距离公式中的p，默认值为2，计算欧氏距离。
'''
"""
我们可以使用网格搜索（Grid Search）和交叉验证（Cross Validation）的方式对模型的超参数进行调整，评估模型的泛化能力，提升模型的预测效果。
网格搜索就是通过穷举法遍历给定的超参数空间，找到最优的超参数组合；
交叉验证则是将训练集分成多个子集，通过在不同的训练集和验证集上进行多次训练和评估，对模型的预测效果进行综合评判。
K-Fold交叉验证是最常用的交叉验证方法，通过将数据集划分为 K 个子集，每次选取其中一个子集作为验证集，剩下的 K-1 个子集作为训练集，对每个子集重复这个过程，完成 K 次训练和评估并将平均值作为模型的最终性能评估
"""
########  scikit-learn 库中的GridSearchCV来做网格搜索和交叉验证，通过这种方式找到针对鸢尾花数据集实施 kNN 算法的最优参数。########
from sklearn.model_selection import GridSearchCV

# 网格搜索交叉验证
gs = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid={
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    cv=5
)
gs.fit(X_train, y_train)
# 获得最优参数及其评分
print('最优参数:', gs.best_params_)
print('评分:', gs.best_score_)
print(gs.predict(X_test))


######### 二、KNN回归算法 ##########

# 每月收入
incomes = np.array([
    9558, 8835, 9313, 14990, 5564, 11227, 11806, 10242, 11999, 11630,
    6906, 13850, 7483, 8090, 9465, 9938, 11414, 3200, 10731, 19880,
    15500, 10343, 11100, 10020, 7587, 6120, 5386, 12038, 13360, 10885,
    17010, 9247, 13050, 6691, 7890, 9070, 16899, 8975, 8650, 9100,
    10990, 9184, 4811, 14890, 11313, 12547, 8300, 12400, 9853, 12890
])
# 每月网购支出
outcomes = np.array([
    3171, 2183, 3091, 5928, 182, 4373, 5297, 3788, 5282, 4166,
    1674, 5045, 1617, 1707, 3096, 3407, 4674, 361, 3599, 6584,
    6356, 3859, 4519, 3352, 1634, 1032, 1106, 4951, 5309, 3800,
    5672, 2901, 5439, 1478, 1424, 2777, 5682, 2554, 2117, 2845,
    3867, 2962,  882, 5435, 4174, 4948, 2376, 4987, 3329, 5002
])
X = np.sort(incomes).reshape(-1, 1)  # 将收入排序后处理成二维数组
y = outcomes[np.argsort(incomes)]    # 将网购支出按照收入进行排序

from sklearn.neighbors import KNeighborsRegressor

# 创建模型
model = KNeighborsRegressor()
# 训练模型
model.fit(X, y)
# 预测结果
y_pred = model.predict(X)

# 原始数据散点图
plt.scatter(X, y, color='navy')
# 预测结果折线图
plt.plot(X, y_pred, color='coral')
plt.show()