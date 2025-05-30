import numpy as np
from scipy import stats
import heapq
import statistics
import random
# 每月收入
x = [9558, 8835, 9313, 14990, 5564, 11227, 11806, 10242, 11999, 11630,
     6906, 13850, 7483, 8090, 9465, 9938, 11414, 3200, 10731, 19880,
     15500, 10343, 11100, 10020, 7587, 6120, 5386, 12038, 13360, 10885,
     17010, 9247, 13050, 6691, 7890, 9070, 16899, 8975, 8650, 9100,
     10990, 9184, 4811, 14890, 11313, 12547, 8300, 12400, 9853, 12890]
# 每月网购支出
y = [3171, 2183, 3091, 5928, 182, 4373, 5297, 3788, 5282, 4166,
     1674, 5045, 1617, 1707, 3096, 3407, 4674, 361, 3599, 6584,
     6356, 3859, 4519, 3352, 1634, 1032, 1106, 4951, 5309, 3800,
     5672, 2901, 5439, 1478, 1424, 2777, 5682, 2554, 2117, 2845,
     3867, 2962,  882, 5435, 4174, 4948, 2376, 4987, 3329, 5002]

sample_data = {key: value for key, value in zip(x, y)}
#kNN算法简单实现
def predict_by_knn(history_data, param_in, k=5) :
    """
    :param history_data: 历史数据
    :param param_in: 模型输入
    :param k: 邻居数量（默认为5）
    :return: 模型输出（预测值）
    """
    neighbors = heapq.nsmallest(k, history_data, key=lambda x: (x-param_in)**2)
    return statistics.mean(history_data[neighbor] for neighbor in neighbors)

incomes = [1800, 3500, 5200, 6600, 13400, 17800, 20000, 30000]
for income in incomes:
    print(f'月收入: {income:>5d}元, 月网购支出: {predict_by_knn(sample_data, income):>6.1f}元')

#定义损失函数，用蒙特卡洛方法随机取值得到最小损失函数
def get_loss(X_, y_, a_, b_):
    """损失函数
    :param X_: 回归模型的自变量
    :param y_: 回归模型的因变量
    :param a_: 回归模型的斜率
    :param b_: 回归模型的截距
    :return: MSE（均方误差）
    """
    y_hat = [a_ * x + b_ for x in X_]
    return statistics.mean([(v1 - v2) ** 2 for v1, v2 in zip(y_, y_hat)])

# 先将最小损失定义为一个很大的值
min_loss, a, b = 1e12, 0, 0

for _ in range(100000):
    # 通过产生随机数的方式获得斜率和截距
    _a, _b = random.random(), random.random() * 4000 - 2000
    # 带入损失函数计算回归模型的MSE
    curr_loss = get_loss(x, y, _a, _b)
    if curr_loss < min_loss:
        # 找到更小的MSE就记为最小损失
        min_loss = curr_loss
        # 记录下当前最小损失对应的a和b
        a, b = _a, _b

print(f'MSE = {min_loss}')
print(f'{a = }, {b = }')


#numpy库函数求解
a, b = np.polyfit(x, y, 1)
print(f'{a = }, {b = }')