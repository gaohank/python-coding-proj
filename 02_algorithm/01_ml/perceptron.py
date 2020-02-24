from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np


# 感知机——线性分类器，只有一个神经元

def sgn(x):
    return np.array([1 if elem >= 0 else 0 for elem in x])[:, np.newaxis]


def train(data, label):
    n_samples, n_features = data.shape
    loop = 100
    alpha = 0.01
    X = np.column_stack((data, np.ones((n_samples, 1))))
    W = np.zeros((n_features + 1, 1))
    for i in range(loop):
        y_predict = sgn(X * W)
        delta_w = X.T * (label.T - y_predict)
        W = W + alpha * delta_w

    return np.delete(W, -1, axis=0), W[-1]


def predict(x, w, b):
    return sgn(np.dot(x, w) + b)


if __name__ == '__main__':
    # x = np.mat([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # y = np.mat([[0], [0], [0], [1]])
    # w, b = train(x, y)
    # print(w, b)
    # print(predict([1, 1, 1], w, b))
    x, y = make_blobs(centers=2, n_samples=100, n_features=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=8)
    w, b = train(np.mat(x_train), np.mat(y_train))
    print(w, b)
    y_p_train = predict(np.mat(x_train), w, b)
    print(y_p_train - y_train)
    print(f"training accuracy: {100 - np.mean(np.abs(y_p_train - y_train)) * 100}%")
