import mxnet as mx
import numpy as np

# construct a simple MLP
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=64)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
out = mx.symbol.SoftmaxOutput(fc3, name='softmax')

# construct the module
mod = mx.mod.Module(out)

print(mod.symbol)

# mod.bind(data_shapes=train_dataiter.provide_data,
#          label_shapes=train_dataiter.provide_label)
#
# mod.init_params()
# mod.fit(train_dataiter, eval_data=eval_dataiter,
#         optimizer_params={'learning_rate': 0.01, 'momentum': 0.9},
#         num_epoch=n_epoch)