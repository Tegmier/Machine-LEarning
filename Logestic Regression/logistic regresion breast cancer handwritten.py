import pandas as pd
import numpy as np
import functions
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("./Logestic Regression/data/breast_cancer.csv")

data["Class"] = data["Class"].apply(lambda x: 0 if x ==2 else 1)

data_train = data.sample(frac=0.8, random_state=42)
data_test = data.drop(data_train.index)

########################### normalization by SKLearn ###############################
# data = pd.read_csv("./Logestic Regression/data/breast_cancer.csv")
# data["Class"][data["Class"] == 2] = 0
# data["Class"][data["Class"] == 4] = 1
# X = data.drop(columns = "Class")
# Y = data["Class"]
# X = StandardScaler().fit_transform(X)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

########################### normalization by Tegmier ###############################
data = pd.read_csv("./Logestic Regression/data/breast_cancer.csv")
data["Class"][data["Class"] == 2] = 0
data["Class"][data["Class"] == 4] = 1
data_train = data.sample(frac=0.8, random_state=42)
data_test = data.drop(data_train.index)

x_train = data_train.drop(columns = "Class").values
y_train = data_train["Class"].values
x_test = data_test.drop(columns = "Class").values
y_test = data_test["Class"].values

x_train_1, y_train_1 = functions.normalization(x_train, y_train)
x_test_1, y_test_1 = functions.normalization(x_test, y_test)

#######################################################################################
print(x_train[0], x_train_1[0])

w = np.zeros(9)
b = 0
learning_rate = 0.001
lambda_rate = 0.01
loss = []
w_history = []
b_history = []

for i in range(10000000):
    loss.append(functions.cal_loss(x_train, y_train, w, b))
    w_history.append(w[0])
    b_history.append(b)
    w,b = functions.update(x_train, y_train, w, b, learning_rate, lambda_rate)

print(w)
y_predict = functions.predict(x_test, w, b)
print("Learning Accuracy is: " + str(functions.learning_accuracy(y_test, y_predict)))

plt.plot(loss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("The Loss Change")
plt.show()







