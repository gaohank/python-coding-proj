from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 得到训练集和测试集
x, y = make_blobs(centers=3, n_samples=100, n_features=3)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=8)


x,y = make_classification(100, 3, n_informative=2, n_redundant=0, n_repeated=0,  n_classes=2)
print(x)

