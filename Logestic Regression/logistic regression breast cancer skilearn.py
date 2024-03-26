import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("./Logestic Regression/data/breast_cancer.csv")

X = data.drop(columns = "Class")
Y = data["Class"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(type(X_train))
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# 创建逻辑回归模型
model = LogisticRegression()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算模型在测试集上的准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 打印分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))
