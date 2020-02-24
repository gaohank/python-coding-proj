import random
import numpy as np


def sigmoid(value):
    return 1.0 / (1 + np.exp(-value))


def getW(data_matrix, label_matrix):
    m, n = np.shape(data_matrix)
    alpha = 0.1
    max_loop = 100
    # 构建列向量
    weights = np.ones((n, 1))
    for value in range(max_loop):
        index = random.randint(0, m - 1)
        y_pre = sigmoid(data_matrix[index] * weights)
        grad = data_matrix[index].transpose() * (label_matrix[index] - y_pre)
        weights = weights + alpha * grad
    return weights


if __name__ == '__main__':
    x = np.mat([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.mat([[0], [0], [0], [1]])
    w = getW(x, y)
    print(w)
    x_w = x * w
    for i in range(x_w.size):
        print(sigmoid(x_w[i]))
