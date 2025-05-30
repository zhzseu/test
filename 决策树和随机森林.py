# 计算信息熵、信息增益、信息增益比，描述特征的数据划分能力
import numpy as np

def entropy(y) :
    """
    计算信息熵
    :param y: 数据集目标值
    :return: 信息熵
    """
    _, counts = np.unique(y, return_counts = True)
    prob = counts / y.size
    return -np.sum(prob * np.log2(prob))
'''
信息增益是在得到特征 A 的信息后，数据集 D 的不确定性减少的程度。
换句话说，信息增益是一种描述数据集确定性增加的量，特征的信息增益越大，
特征的分类能力就越强，在给定该特征后数据集的确定性就越大。
'''
def info_gain(x, y):
    """
    计算信息增益
    :param x: 给定的特征
    :param y: 数据集的目标值
    :return: 信息增益
    """
    values, counts = np.unique(x, return_counts = True)
    new_entropy = 0
    for i, value in enumerate(values):
        prob = counts[i] / x.size
        new_entropy += prob * entropy(y[x ==value])
    return entropy(y) - new_entropy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#### 计算鸢尾花数据集四个特征的信息增益
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)
print(f'H(D)    = {entropy(y_train)}')
print(f'g(D,A0) = {info_gain(X_train[:, 0],y_train)}')
print(f'g(D,A1) = {info_gain(X_train[:, 1],y_train)}')
print(f'g(D,A2) = {info_gain(X_train[:, 2],y_train)}')
print(f'g(D,A3) = {info_gain(X_train[:, 3],y_train)}')

'''
当某个特征取值较多时，该特征的信息增益计算结果就会比较大，所以使用信息增益选择特征时，
会偏向于取值较多的特征。为了解决这个问题，我们可以计算信息增益比
'''
def info_gain_ratio(x, y):
    """
    计算信息增益比
    :param x: 给定的特征
    :param y: 数据集的目标值
    :return: 信息增益比
    """
    return info_gain(x, y) / entropy(x)
print()
print(f'R(D,A0) = {info_gain_ratio(X_train[:, 0], y_train)}')
print(f'R(D,A1) = {info_gain_ratio(X_train[:, 1], y_train)}')
print(f'R(D,A2) = {info_gain_ratio(X_train[:, 2], y_train)}')
print(f'R(D,A3) = {info_gain_ratio(X_train[:, 3], y_train)}')

"""
信息增益比越大，不确定度越大，该特征的数据划分能力越强，可以作为决策树根节点
"""

# 计算基尼指数选择特征，纯度越高Gini越趋近于0，反之趋近于1
# 如果数据集有n个类别，样本属于第k个类别的概率为 p[k]
# 那么数据集的基尼指数可以通过下面的公式进行计算：
# Gini(data) = 1 - sum(p[k]**2)

# 如果数据集 data根据特征A划分为k个部分
# ##那么在给定特征A的前提条件下##
# 数据集的基尼指数可以定义为：
# sum(data[i].size/data.size*Gini(data[i]))
def gini_index(y):
    """
    计算基尼指数
    :param y: 数据集目标值
    :return: 基尼指数
    """
    _, counts = np.unique(y, return_counts = True)
    return 1 - np.sum((counts / y.size) ** 2)

def gini_with_feature(x, y):
    """
    计算给定特征后的基尼指数
    :param x: 给定的特征
    :param y: 数据集的目标值
    :return: 给定特征后的基尼指数
    """
    values, counts = np.unique(x, return_counts = True)
    gini = 0
    for value in values:
        prob = x[x ==value].size / x.size
        gini += prob * gini_index(y[x == value])
    return gini
print(f'G(D)    = {gini_index(y_train)}')
print(f'G(D,A0) = {gini_with_feature(X_train[:, 0], y_train)}')
print(f'G(D,A1) = {gini_with_feature(X_train[:, 1], y_train)}')
print(f'G(D,A2) = {gini_with_feature(X_train[:, 2], y_train)}')
print(f'G(D,A3) = {gini_with_feature(X_train[:, 3], y_train)}')

#决策树模型的实现
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report
print(y_test)
print(y_pred)
print(classification_report(y_test, y_pred))
#可视化决策树
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 10))
plot_tree(
    decision_tree=model,               # 决策树模型
    feature_names=iris.feature_names,  # 特征的名称
    class_names=iris.target_names,     # 标签的名称
    filled=True,

)
plt.show()

# 换用信息增益创建模型
model = DecisionTreeClassifier(
    criterion='entropy',
    ccp_alpha=0.01,

)
# 训练模型
model.fit(X_train, y_train)
# 可视化
plt.figure(figsize=(12, 10))
plot_tree(
    decision_tree=model,  # 决策树模型
    feature_names=iris.feature_names,  # 特征的名称
    class_names=iris.target_names,  # 标签的名称
    filled=True  # 用颜色填充
)
plt.show()
# 模型参数调优
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid={
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(5, 10),
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_leaf': np.arange(1, 11),
        'max_leaf_nodes': np.arange(5, 15)
    },
    cv=5
)
gs.fit(X_train, y_train)
# 随机森林，可能需要运行很久
from sklearn.ensemble import RandomForestClassifier

gs = GridSearchCV(
    estimator=RandomForestClassifier(n_jobs=-1),
    param_grid={
        'n_estimators': [50, 100, 150],
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(5, 10),
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': np.arange(1, 11),
        'max_leaf_nodes': np.arange(5, 15)
    },
    cv=5
)
gs.fit(X_train, y_train)

