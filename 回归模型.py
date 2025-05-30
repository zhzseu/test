import ssl
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv('https://archive.ics.uci.edu/static/public/9/data.csv')

"""
属性名称	                    描述
car_name	    汽车的名称，字符串，这个属性对建模暂时没有帮助
cylinders	    气缸数量，整数
displacement	发动机排量（立方英寸），浮点数
horsepower	    马力，浮点数，有空值需要提前处理
weight	        汽车重量（磅），整数
acceleration	加速（0 - 60 mph所需时间），浮点数
model_year	    模型年份（1970年 - 1982年），这里用的是两位的年份
origin	        汽车来源（1 = 美国, 2 = 欧洲, 3 = 日本），这里的1、2、3应该视为三种类别而不是整数
mpg	            车辆的燃油效率，每加仑行驶的里程（目标变量）
"""

df.info()
# 删除车名，用不上
df.drop(columns=['car_name'], inplace=True)

# 删除有缺失值的样本
df.dropna(inplace=True)
# 将origin字段处理为类别类型
df['origin'] = df['origin'].astype('category')
# 将origin字段处理为独热编码
# 独热编码是把类别变量扩展为由0/1组成的向量，让算法能正确处理类别特征。
df = pd.get_dummies(df, columns=['origin'], drop_first=True)
print(df)

from sklearn.model_selection import train_test_split

X, y = df.drop(columns=['mpg']).values, df['mpg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=3)

# 用scikit-learn库中的linear_model模块的LinearRegression创建线性回归模型，LinearRegression使用最小二乘法计算参数
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("线性回归模型：")
print("回归系数：", model.coef_)
print("截距：", model.intercept_)

# 使用 scikit-learn 中封装好的函数计算出均方误差、平均绝对误差和决定系数的值

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'均方误差: {mse:.4f}')
print(f'平均绝对误差: {mae:.4f}')
print(f'决定系数: {r2:.4f}')

from sklearn.linear_model import Ridge

model = Ridge()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Ridge岭回归，引入一个正则化项L2 = lambda * sum(beta^2), beta为回归系数：")
print('回归系数:', model.coef_)
print('截距:', model.intercept_)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'均方误差: {mse:.4f}')
print(f'决定系数: {r2:.4f}')
print()

# 通过 scikit-learn 库linear_model模块的Lasso类实现套索回归（引入L1正则化项，不仅防止过拟合，还具有特征选择的功，特别适用于高维数据）
from sklearn.linear_model import Lasso

model = Lasso()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Lasso套索回归，引入正则化项L1 = lambda * sum(abs(beta)),将不重要的回归系数缩减成0：")
print('回归系数:', model.coef_)
print('截距:', model.intercept_)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'均方误差: {mse:.4f}')
print(f'决定系数: {r2:.4f}')

print()
########## 梯度下降法求解回归模型参数 ###########

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# 对特征进行选择和标准化处理
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X[:, [1, 2, 3, 5]])
# 重新拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, train_size=0.8, random_state=3)

# 模型的创建、训练和预测
model = SGDRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('回归系数:', model.coef_)
print('截距:', model.intercept_)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'均方误差: {mse:.4f}')
print(f'决定系数: {r2:.4f}')
print()
"""
1.loss：指定优化目标（损失函数），默认值为'squared_error'（最小二乘法），其他可以选择的值有：'huber'、'epsilon_insensitive' 和 'squared_epsilon_insensitive'，其中'huber'适用于对异常值更鲁棒的回归模型。
2.penalty：指定正则化方法，用于防止过拟合，默认为'l2'（L2 正则化），其他可以选择的值有：'l1'（L1正则化）、'elasticnet'（弹性网络，L1 和 L2 的组合）、None（不使用正则化）。
3.alpha：正则化强度的系数，控制正则化项的权重，默认值为0.0001。较大的 alpha 值会加重正则化的影响，从而限制模型复杂度；较小的值会让模型更关注训练数据的拟合。
4.l1_ratio：当 penalty='elasticnet' 时，控制 L1 和 L2 正则化之间的权重，默认值为0.15，取值范围为[0, 1]（0 表示完全使用 L2，1 表示完全使用 L1）。
5.tol：优化算法的容差，即判断收敛的阈值，默认值为1e-3。当目标函数的改变量小于 tol 时，训练会提前终止；如果希望训练更加精确，可以适当降低 tol。
6.learning_rate：指定学习率的调节策略，默认值为'constant'，表示使用固定学习率，具体的值由 eta0 指定；其他可选项包括：
1)'optimal'：基于公式eta = 1.0 / (alpha * (t + t0))自动调整。
2)'invscaling'：按 eta = eta0 / pow(t, power_t) 缩放学习率。
3)'adaptive'：动态调整，误差减少时保持当前学习率，否则减小学习率。
7.eta0：初始学习率，默认值为0.01，当 learning_rate='constant' 或其他策略使用时，eta0 决定了初始更新步长。
8.power_t：当 learning_rate='invscaling' 时，控制学习率衰减速度，默认值为0.25。较小的值会让学习率下降得更慢，从而更长时间地关注全局优化。
9.early_stopping：是否启用早停机制，默认值为False。如果设置为 True，模型会根据验证集性能自动停止训练，防止过拟合。
10.validation_fraction：指定用作验证集的训练数据比例，默认值为0.1。当 early_stopping=True 时，该参数会起作用。
11.max_iter：训练的最大迭代次数，默认值为1000。当数据较大或学习率较小时，可能需要增加迭代次数以保证收敛。
12.shuffle：是否在每个迭代轮次开始时打乱训练数据，默认值为True，表示打乱数据。打乱数据有助于提高模型的泛化能力。
13.warm_start：是否使用上次训练的参数继续训练，默认值为False。当设置为 True 时，可以在已有模型的基础上进一步优化。
14.verbose：控制训练过程的日志输出，默认值为0，可以设置为更高值以观察训练进度。
"""

########## 多项式回归分析 ##########

"""
创建`PolynomialFeatures`对象时有几个重要的参数：

1. `degree`：设置多项式的最高次项。例如，`degree=3` 会生成包含一次项、二次项和三次项的特征。
2. `interaction_only`：默认值为`False`，如果设置为`True`，则只生成交互项（如 $\small{x_{1}x_{2}}$ ），不生成单独的高次项（如 $\small{x_{1}^{2}}$ 、 $\small{x_{2}^{2}}$ ）。
3. `include_bias`：默认值为`True`，表示包括常数项（通常为 1），设置为`False`则不包括常数项。

"""
import numpy as np
import matplotlib.pyplot as plt
# 随机生成多项式离散点
x = np.linspace(0, 6, 150)
y = x ** 2 - 4 * x + 3 + np.random.normal(1, 1, 150)
plt.scatter(x, y)
plt.show()

from sklearn.preprocessing import PolynomialFeatures
x_= x.reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
x_ = poly.fit_transform(x_)

model = LinearRegression()
model.fit(x_, y)
y_pred = model.predict(x_)
r2 = r2_score(y, y_pred)
print("多项式拟合：")
print(f'决定系数: {r2:.4f}')


########## 逻辑回归 ###########

"""
逻辑回归的核心思想是通过 Sigmoid 函数将线性回归的输出映射到(0,1)区间，作为对分类概率的预测。
f(y) = 1/(1+e^y)
"""
# 创建一组模拟数据
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 生成1000条样本数据，每个样本包含6个特征
X, y = make_classification(n_samples=1000, n_features=6, random_state=3)
# 将1000条样本数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)

# 创建和训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 对测试集进行预测并评估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

"""
逻辑回归模型比较重要的超参数，如下所示：

1. `penalty`：指定正则化类型，用于控制模型复杂度，防止过拟合，默认值为`L2`。
2. `C`：正则化强度的倒数，默认值为`1.0`。较小的 `C` 值会加强正则化（更多限制模型复杂度），较大的 `C` 值会减弱正则化（更注重拟合训练数据）。
3. `solver`：指定优化算法，默认值为`lbfgs`，可选值包括：
    - `'newton-cg'`、`'lbfgs'`、`'sag'`、`'saga'`：支持 L2 和无正则化。
    - `'liblinear'`：支持 L1 和 L2 正则化，适用于小型数据集。
    - `'saga'`：支持 L1、L2 和 ElasticNet，适用于大规模数据。
4. `multi_class`：指定多分类问题的处理方式，默认值为`'auto'`，根据数据选择 `'ovr'` 或 `'multinomial'`，前者表示一对多策略，适合二分类或多分类的基础情况，后者表示多项式回归策略，适用于多分类问题，需与 `'lbfgs'`、`'sag'` 或 `'saga'` 搭配使用。
5. `fit_intercept`：是否计算截距（偏置项），默认值为`True`。
6. `class_weight`：类别权重，处理类别不平衡问题，默认值为`None`，设置为`'balanced'`可以根据类别频率自动调整权重。

"""
