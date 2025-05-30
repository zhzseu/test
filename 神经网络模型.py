# -*- coding: utf-8 -*-
"""
神经网络模型基础示例与实现

本文件包含：
1. 神经网络基本原理与结构说明
2. 使用 scikit-learn MLPClassifier 实现分类
3. 使用 PyTorch 实现分类
4. 使用 PyTorch 实现回归

知识点均以注释形式穿插代码，便于学习和理解
"""

############################################################
# 一、神经网络基础知识
############################################################
"""
神经网络是一种模拟人脑神经元连接和工作方式的机器学习算法，其基本结构包括输入层、隐藏层和输出层。
- 输入层：接收输入数据，每个神经元对应一个特征
- 隐藏层：可以有多层，负责特征提取和变换
- 输出层：根据任务类型输出最终结果（分类/回归）

神经元的计算公式：
y = f(Σ(w_ix_i) + b)
其中，w是权重，b是偏置，f是激活函数，常见激活函数有Sigmoid、Tanh、ReLU、Leaky ReLU等。

神经网络训练过程通常使用反向传播算法和梯度下降法，通过最小化损失函数来不断调整权重和偏置。
"""

############################################################
# 二、scikit-learn 实现多层感知机分类器 (MLPClassifier)
############################################################
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 1. 加载数据集（Iris 鸢尾花数据集）
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)

# 3. 构建多层感知机模型
# 参数说明：
# - solver: 优化器选择，如 'lbfgs', 'sgd', 'adam'
# - activation: 激活函数，如 'relu', 'tanh', 'logistic'
# - hidden_layer_sizes: 每层神经元数量 (如 (32,32,32) 表示有3层，每层32个神经元)
model = MLPClassifier(
    solver='lbfgs',
    learning_rate='adaptive',
    activation='relu',
    hidden_layer_sizes=(32, 32, 32)  # 比较深的网络结构，效果优于单层
)
# 4. 训练模型
model.fit(X_train, y_train)
# 5. 预测与评估
y_pred = model.predict(X_test)
print("#" * 60)
print("scikit-learn MLPClassifier 分类评估报告：")
print(classification_report(y_test, y_pred))

"""
结论：
- hidden_layer_sizes 设置为 (1,) 时准确率较低，设置为 (32,32,32) 能大幅提升准确率
- 可以通过调整超参数改善模型效果
"""

############################################################
# 三、PyTorch 实现神经网络分类（鸢尾花数据集）
############################################################
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 数据预处理（标准化有助于提升训练效率和模型表现）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, random_state=3)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 2. 定义神经网络结构
class IrisNN(nn.Module):
    """鸢尾花神经网络模型"""
    def __init__(self):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)  # 输入4维特征，隐藏层32个神经元
        self.fc2 = nn.Linear(32, 3)  # 输出3类

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 隐藏层激活函数ReLU
        x = self.fc2(x)
        return x

# 3. 实例化模型、损失函数与优化器
model = IrisNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
for epoch in range(256):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_function(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 32 == 0:
        print(f"Epoch {epoch+1}/256, Loss: {loss.item():.4f}")

# 5. 模型评估
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    _, y_pred_tensor = torch.max(logits, 1)
    print("#" * 60)
    print("PyTorch 神经网络分类准确率: {:.2%}".format(accuracy_score(y_test_tensor, y_pred_tensor)))
    print("分类评估报告：")
    print(classification_report(y_test_tensor, y_pred_tensor))

"""
知识点总结：
- 神经网络模型可以灵活调整层数和每层神经元数量
- 常用激活函数有 ReLU、Sigmoid、Tanh
- 训练时采用交叉熵损失和Adam优化器
"""

############################################################
# 四、PyTorch 实现神经网络回归（汽车MPG数据集）
############################################################
import ssl
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# 1. 解决SSL证书问题（部分国内网络环境下需要）
ssl._create_default_https_context = ssl._create_unverified_context

def load_prep_data():
    """加载并准备汽车MPG数据"""
    df = pd.read_csv('https://archive.ics.uci.edu/static/public/9/data.csv')
    # 数据清理与预处理
    df.drop(columns=['car_name'], inplace=True)
    df.dropna(inplace=True)
    df['origin'] = df['origin'].astype('category')
    df = pd.get_dummies(df, columns=['origin'], drop_first=True).astype('f8')
    scaler = StandardScaler()
    return scaler.fit_transform(df.drop(columns='mpg').values), df['mpg'].values

# 2. 定义回归神经网络
class MLPRegressor(nn.Module):
    """回归用神经网络"""
    def __init__(self, n_features):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main_regression():
    # 加载数据
    X, y = load_prep_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 实例化模型、损失函数和优化器
    model = MLPRegressor(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    epochs = 256
    for epoch in range(epochs):
        model.train()
        y_pred_tensor = model(X_train_tensor)
        loss = criterion(y_pred_tensor, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 16 == 0:
            print(f'Epoch [{epoch + 1} / {epochs}], Loss: {loss.item():.4f}')

    # 测试
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = mean_squared_error(y_test, y_pred.numpy())
        r2 = r2_score(y_test, y_pred.numpy())
    print("#" * 60)
    print(f'Test MSE: {test_loss:.4f}')
    print(f'Test R2: {r2:.4f}')

if __name__ == '__main__':
    # 运行回归实验（如不需要可注释掉）
    main_regression()

"""
本文件总结：
- 神经网络模型以多层结构拟合复杂关系，常用于分类和回归任务
- 可用 scikit-learn 快速实现MLP分类器
- PyTorch 框架下可灵活自定义神经网络结构，适用于更复杂的任务
- 实验表明，合适的网络结构和超参数选择对模型表现有决定性影响
- 神经网络对数据规模和质量较为敏感，实际应用中需关注数据预处理与模型调优
"""