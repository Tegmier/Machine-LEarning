import pandas as pd
import numpy as np
import functions
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("./Logestic Regression/data/breast_cancer.csv")
x1 = data.drop(columns="Class").head(4)
x2 = x1.copy().values
y2 = data["Class"].head(4)
x1 = StandardScaler().fit_transform(x1)

x2, y2 = functions.normalization(x2, y2)

print(x1)
print(x2)