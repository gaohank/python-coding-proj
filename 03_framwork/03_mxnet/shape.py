import mxnet as mx

a = mx.sym.Variable('A')
b = mx.sym.Variable('B')
c = (a + b) / 10
d = c + 1
input_shapes = {'A': (10, 2), 'B': (10, 2)}  # 定义输入的shape
arg_shapes, out_shapes, aux_shapes = d.infer_shape(**input_shapes)

print(arg_shapes)
print(out_shapes)
print(aux_shapes)
