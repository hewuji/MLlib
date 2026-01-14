import numpy as np
from keras.datasets import boston_housing


(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print("X_train的形状：", X_train.shape)
print("X_train中第⼀个样本的形状：", X_train[0].shape)
print("y_train的形状：", y_train.shape)


weight = np.array([1, -1.8, 1, 1, 2])
X = np.array([1, 6, 7, 8, 9])
y_hat = np.dot(X, weight)
print('函数返回结果：', y_hat) #输出预测结果