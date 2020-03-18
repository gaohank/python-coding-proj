import mxnet as mx
import numpy as np

v1 = np.array([[1, 2]])
v2 = np.array([[0, 1]])
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
b_stop_grad = 4 * b
a_stop_grad = 3 * a
loss = mx.sym.MakeLoss(b_stop_grad + a_stop_grad)
executor = loss.simple_bind(ctx=mx.cpu(), a=(1, 2), b=(1, 2))
executor.forward(is_train=True, a=v1, b=v2)
executor.backward()
print(executor.grad_arrays)
