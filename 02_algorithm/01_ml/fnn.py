import numpy as np


# 前馈神经网络

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(x, W_o, W_h):
    A_h = np.dot(x, W_h)
    O_h = np.tanh(A_h)

    # Compute activations and outputs of output units
    A_o = np.dot(O_h, W_o)
    O_o = sigmoid(A_o)

    outputs = {
        "A_h": A_h,
        "A_o": A_o,
        "O_h": O_h,
        "O_o": O_o,
    }

    return outputs


def cost(y, y_predict, n_samples):
    cost = (-1 / n_samples) * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
    # 删除单维度条目
    return np.squeeze(cost)


def backward(x, y, n_samples, outputs, W_o):
    dA_o = (outputs["O_o"] - y)
    dW_o = (1 / n_samples) * np.dot(outputs["O_h"].T, dA_o)
    dA_h = (np.dot(dA_o, W_o.T)) * (1 - np.power(outputs["O_h"], 2))
    dW_h = (1 / n_samples) * np.dot(x.T, dA_h)
    gradients = {"dW_o": dW_o, "dW_h": dW_h}
    return gradients


def update_weights(gradients, alpha, W_o, W_h):
    W_o = W_o - alpha * gradients["dW_o"]
    W_h = W_h - alpha * gradients["dW_h"]


def train(data, label, loop=100, alpha=0.1):
    n_samples, n_features = data.shape
    X = np.column_stack((data, np.ones((n_samples, 1))))
    n_inputs = 3
    n_outputs = 2
    hidden = 1
    # 可能有负值，且服从正太分布，rand获得0-1的随机数
    W_h = np.random.randn(n_inputs, hidden)
    W_o = np.random.randn(hidden, n_outputs)
    for i in loop:
        outputs = forward(X, W_o, W_h)
        cost(label, outputs["O_o"], n_samples=n_samples)
        gradients = backward(X, label, n_samples, outputs, W_o)
        # if i % 100 == 0:
        #     print(f'Cost at iteration {i}: {np.round(cost, 4)}')
        update_weights(gradients, alpha, W_o, W_h)
