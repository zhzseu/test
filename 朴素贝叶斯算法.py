from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
# 注意：test_size=0.8 意味着只有 20% 的数据用于训练，这在实际应用中可能过少
# 这里保持原样以复现问题，但通常会使用更大的训练集比例（如 0.2 或 0.3 作为 test_size）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

def naive_bayes_fit(X, y):
    """
    朴素贝叶斯分类器训练函数
    :param X: 样本特征 (NumPy 数组)
    :param y: 样本标签 (NumPy 数组)
    :return: 二元组 - (先验概率字典, 似然性(条件概率)字典)
    """
    # 创建 X 的副本，以避免修改原始数据
    # pd.cut 会返回 Categorical 对象，直接赋值到 NumPy 数组的切片可能会导致类型问题
    # 因此，我们确保 X 保持为 NumPy 数组，并正确处理 pd.cut 的结果
    X_processed = np.copy(X)

    # 计算先验概率 P(C_i)
    class_labels, class_counts = np.unique(y, return_counts=True)
    # 使用 Pandas Series 存储先验概率，键为类别标签，值为概率
    prior_probs = pd.Series({k: v / y.size for k, v in zip(class_labels, class_counts)})

    # 保存似然性计算结果字典 P(X_j | C_i)
    likelihoods = {}

    # 对每个特征进行循环
    for j in range(X_processed.shape[1]):
        # 对特征进行等宽分箱（离散化处理）
        # pd.cut 返回 Categorical 对象，其 categories 是标签
        # .codes 属性可以获取这些标签对应的整数编码
        # 我们需要确保这些标签是可用于索引的
        binned_feature = pd.cut(X_processed[:, j], bins=5, labels=np.arange(1, 6), include_lowest=True)
        # 将分箱结果转换为 NumPy 数组，确保后续操作的兼容性
        X_processed[:, j] = binned_feature.astype(int) # 转换为整数类型

        # 对每个类别进行循环，计算该类别下每个特征值的似然性
        for i in prior_probs.index: # i 代表类别标签
            # 按标签类别拆分数据，获取当前特征的子集
            x_prime = X_processed[y == i, j]

            # 统计当前类别下，当前特征的每个离散值出现的频次
            # x_values 包含唯一的特征值 (e.g., 1, 2, 3, 4, 5)
            # x_counts 包含这些特征值对应的频次
            x_values, x_counts = np.unique(x_prime, return_counts=True)

            # 遍历每个唯一的特征值及其对应的频次
            # k 现在是特征值，count 是该特征值的频次
            for k_feature_value, count in zip(x_values, x_counts):
                # 计算似然性 P(X_j = k_feature_value | C_i = i)
                # 并保存在字典中，字典的键是一个三元组 - (类别标签, 特征序号, 特征值)
                # 这里的 k_feature_value 是分箱后的特征值 (1到5)
                likelihoods[(i, j, k_feature_value)] = count / x_prime.size
    return prior_probs, likelihoods

# 调用训练函数
p_ci, p_x_ci = naive_bayes_fit(X_train, y_train)

print('先验概率:')
print(p_ci)
print('\n似然性:')
print(p_x_ci)

def naive_bayes_predict(X, p_ci, p_x_ci):
    """
    朴素贝叶斯分类器预测
    :param X: 样本特征 (NumPy 数组)
    :param p_ci: 先验概率 (Pandas Series)
    :param p_x_ci: 似然性 (字典)
    :return: 预测标签 (NumPy 数组)
    """

    # 1. 创建 X 的副本，并对其进行分箱处理
    # 这一步必须在 X_processed 上进行，并且要确保类型转换
    X_processed = np.copy(X)
    for j in range(X_processed.shape[1]): # 注意这里是 X_processed.shape[1]
        binned_feature = pd.cut(X_processed[:, j], bins=5, labels=np.arange(1, 6), include_lowest=True)
        # 核心修正：分箱后必须转换为整数类型，才能与训练时p_x_ci的键匹配
        X_processed[:, j] = binned_feature.astype(int)

    # results 存储每个样本属于每个类别的后验概率（未归一化）
    # X_processed.shape[0] 是样本数量，p_ci.size 是类别数量
    results = np.zeros((X_processed.shape[0], p_ci.size))
    # 获取类别标签的数组，用于索引
    clazz_labels = p_ci.index.values

    # 遍历每一个待预测的样本
    for k in range(X_processed.shape[0]): # k 代表当前样本的索引
        # 获取当前样本的所有处理后的特征值
        current_sample_features = X_processed[k, :]

        # 遍历每一个可能的类别
        for i, class_label in enumerate(clazz_labels): # i 是类别标签在 clazz_labels 中的索引, class_label 是实际的类别标签 (0, 1, 2)
            # 根据贝叶斯公式 P(C|X) = P(C) * P(X|C) / P(X)
            # 我们只需要计算 P(C) * P(X|C)，因为 P(X) 对于所有类别是常数，不影响argmax
            # 初始化当前类别的后验概率，从先验概率 P(C_i) 开始
            prob = p_ci.loc[class_label] # 使用 class_label 直接从 Series 中获取先验概率

            # 遍历当前样本的每一个特征
            for j in range(X_processed.shape[1]): # j 代表当前特征的索引
                # 获取当前样本、当前特征的离散化后的值
                feature_value_at_j = current_sample_features[j]

                # 核心修正：正确构造查找 p_x_ci 的键
                # 键应该是 (实际类别标签, 特征序号, 当前样本的特征值)
                # 如果找不到对应的似然性（即训练集中该类别下未出现此特征值），
                # 默认为 0。在实际应用中，为了避免 0 导致整个概率乘积为 0，
                # 通常会使用拉普拉斯平滑（加1平滑）或其他平滑方法。
                # 但根据您当前代码的逻辑，此处继续使用 0 作为默认值。
                likelihood = p_x_ci.get((class_label, j, feature_value_at_j), 0)

                # 累乘似然性 P(X_j | C_i)
                prob *= likelihood

            # 将计算得到的（未归一化的）后验概率存储起来
            results[k, i] = prob

    # 返回每个样本概率最大的类别标签
    # argmax(axis=1) 找出每行（每个样本）最大值的索引
    # 然后使用 clazz_labels 将索引转换为实际的类别标签
    return clazz_labels[results.argmax(axis=1)]

# 运行修正后的预测函数
y_pred = naive_bayes_predict(X_test, p_ci, p_x_ci)
print("预测结果是否与真实值匹配 (True/False):")
print(y_pred == y_test)
print("\n预测准确率:")
print(np.mean(y_pred == y_test))
y_pred = naive_bayes_predict(X_test, p_ci, p_x_ci)
print(y_pred == y_test)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))

# 每个样本对用到每个标签给出的概率值

print(model.predict_proba(X_test).round(2))

'''

sklearn库中给出的 5 个朴素贝叶斯算法的变体：
分类器	        特征类型	        主要假设
BernoulliNB	    二元特征	    特征服从 Bernoulli 分布
CategoricalNB	类别特征	    特征服从多项式分布，常用于处理类别数据
ComplementNB	计数特征	    利用补集概率，常用于处理不平衡数据集
GaussianNB	    连续特征	    特征服从高斯分布
MultinomialNB	计数特征	    特征服从多项式分布，常用于文本分类

'''

'''
算法优缺点：

逻辑简单容易实现，适合大规模数据集。
运算开销较小。预测需要用到的概率在训练阶段都已经准好了，当新数据来了之后，只需要获取对应的概率值并进行简单的运算就能获得预测的结果。
受噪声和无关属性影响小。
当然，由于做了“特征相互独立”这个假设，朴素贝叶斯算法的缺点也相当明显，因为在实际应用中，特征之间很难做到完全独立，
尤其是维度很高的数据，如果特征之间的相关性较大，那么分类的效果就会变得很差。
为了解决这个问题，在朴素贝叶斯算法的基础上又衍生出了一些新的方法，包括：
半朴素贝叶斯（One Dependent Estimator）、AODE（Averaged One Dependent Estimator）、K 依赖朴素贝叶斯、朴素贝叶斯网络、高斯混合朴素贝叶斯等

以下是一些常见的应用场景：
文本分类：如垃圾邮件检测、情感分析等。
推荐系统：根据用户行为和喜好进行个性化推荐。
医药诊断：根据症状预测疾病。
'''