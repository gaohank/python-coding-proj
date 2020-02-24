import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 求距离
def euclidean_distance(data, X):
    # axis==1表示横轴，方向从左到右，axis==0表示纵轴，方向从上到下
    # 如果是1维
    if X.ndim == 1:
        l2 = np.sqrt(np.sum((data - X) ** 2, axis=1))

    if X.ndim == 2:
        n_samples, _ = X.shape
        l2 = [np.sqrt(np.sum((data - X[i]) ** 2, axis=1)) for i in range(n_samples)]

    return np.array(l2)


# 输入训练集和测试集，默认选取最近的一个点分类
def predict(x_train, y_train, x, k=1):
    # 计算输入和训练数据的距离
    dists = euclidean_distance(x_train, x)

    # 找到k个相邻节点，默认1个
    if x.ndim == 1:
        if k == 1:
            nn = np.argmin(dists)
            return y_train[nn]
        else:
            # 先排序，取k个临近的点
            knn = np.argsort(dists)[:k]
            # 取k个点对于的类
            y_knn = y_train[knn]
            # 先计算list(y_knn).count()，出现类别最多的，就是该类
            max_vote = max(y_knn, key=list(y_knn).count)
            return max_vote

    if x.ndim == 2:
        # 得到每个样本最近的位置
        knn = np.argsort(dists)[:, :k]
        y_knn = y_train[knn]
        if k == 1:
            return y_knn.T
        else:
            n_samples, _ = x.shape
            max_votes = [max(y_knn[i], key=list(y_knn[i]).count) for i in range(n_samples)]
            return max_votes


digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train)
print(y_train)
print(X_train.shape, X_train.ndim, X_train.dtype)
print(y_train.shape, y_train.ndim, y_train.dtype)
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

print("Testing one datapoint, k=1")
print(f"Predicted label: {predict(X_train, y_train, X_test[0], k=1)}")
print(f"True label: {y_test[0]}")

print("Testing one datapoint, k=5")
print(f"Predicted label: {predict(X_train, y_train, X_test[20], k=5)}")
print(f"True label: {y_test[20]}")

print("Testing 10 datapoint, k=4")
print(f"Predicted labels: {predict(X_train, y_train, X_test[5:15], k=4)}")
print(f"True labels: {y_test[5:15]}")

# Compute accuracy on test set
y_p_test1 = predict(X_train, y_train, X_test, k=1)
test_acc1 = np.sum(y_p_test1[0] == y_test) / len(y_p_test1[0]) * 100
print(f"Test accuracy with k = 1: {format(test_acc1)}")

y_p_test5 = predict(X_train, y_train, X_test, k=5)
test_acc5 = np.sum(y_p_test5 == y_test) / len(y_p_test5) * 100
print(f"Test accuracy with k = 5: {format(test_acc5)}")

fig = plt.figure(figsize=(10, 8))
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    plt.imshow(X[i].reshape((8, 8)), cmap='gray')

plt.show()
