import random
import numpy as np


def softmax(scores):
    """
    :param scores:
    :return:
    """
    exp = np.exp(scores)
    # 按行方向计算，输入5*3的矩阵，得到5*1结果
    sum_exp = np.sum(np.exp(scores), axis=1)
    return exp / sum_exp


def compute_scores(X, weights):
    return np.dot(X, weights.T)


def one_hot(y, n_samples, n_classes):
    """
    Tranforms vector y of labels to one-hot encoded matrix
    n_samples: 样本数量
    n_classes: 分类数量
    """
    arrays = np.zeros((n_samples, n_classes))
    # 将矩阵对应的结果位设置为1
    arrays[np.arange(n_samples), y.T] = 1
    return arrays


def train(data_matrix, label_matrix, n_classes):
    # 得到输入样本矩阵的样本数m，和特征数n
    m, n = np.shape(data_matrix)
    alpha = 0.1
    max_loop = 100
    # 构建权重矩阵，行表示分类，列表示特征
    weights = np.ones((n_classes, n))
    for i in range(max_loop):
        # 通过x * w^T，得到m * n_class的矩阵
        scores = compute_scores(data_matrix, weights)
        # 带入softmax模型，计算预测值，得到m * n_class的对应m个样本的预测值
        probs = softmax(scores)
        #
        y_one_hot = one_hot(label_matrix, m, n_classes)

        dw = (1 / m) * np.dot(data_matrix.T, (probs - y_one_hot))
        weights = weights - alpha * dw

    return weights


if __name__ == '__main__':
    x = np.mat([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.mat([[0], [1], [2], [2], [1]])
    weights = train(x, y, 3)
    print(weights)
    # X, y_true = make_blobs(centers=4, n_samples=50)
    # print(X, y_true)
    # print(np.dot([0, 1, 0], weights))
