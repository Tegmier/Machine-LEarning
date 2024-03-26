import pandas as pd
import numpy as np
def normalization(matrix, y):
    row_means = np.mean(matrix, axis=0, keepdims=True)
    row_stds = np.std(matrix, axis=0, keepdims=True)          
    # standardized_matrix = (matrix - row_means) / row_stds
    # rowstodelete = (row_stds != 0).flatten()
    # standardized_matrix = standardized_matrix[rowstodelete]
    # y = y[rowstodelete]
    row_stds[row_stds == 0] = 1
    standardized_matrix = (matrix - row_means) / row_stds
    return standardized_matrix, y

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def update(x_train, y_train, w, b, learning_rate, lambda_rate):
    part1 = 1 - learning_rate*lambda_rate/len(x_train)
    part2 = np.dot((sigmoid(np.dot(x_train, w) + b) - y_train).T, x_train)
    w = part1 * w - learning_rate * part2 / len(x_train)
    b = b - learning_rate * np.mean(sigmoid(np.dot(x_train, w) + b) - y_train)
    return w,b

def cal_loss(x_train, y_train, w, b):
    return  0 - np.dot(y_train.T, np.log(sigmoid(np.dot(x_train, w) + b))) - np.dot((1 - y_train).T, np.log(1 - sigmoid(np.dot(x_train, w) + b)))

def predict(x_test, w, b):
    predict = sigmoid(np.dot(x_test, w) + b)
    y_predict = np.vectorize(lambda x: 1 if x > 0.5 else 0)(predict)
    return y_predict

def learning_accuracy(y_test, y_predict):
    return np.sum(y_test == y_predict)/len(y_test)