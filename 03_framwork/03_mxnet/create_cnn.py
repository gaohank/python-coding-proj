import sys
import os
import mxnet as mx


def Act(data, act_type, name):
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body


def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data,
                              num_filter=num_filter,
                              kernel=kernel,
                              num_group=num_group,
                              stride=stride,
                              pad=pad,
                              no_bias=True,
                              name='%s%s_conv2d' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv,
                          name='%s%s_batchnorm' % (name, suffix),
                          fix_gamma=False,
                          momentum=0.9)
    act = Act(data=bn, act_type='prelu', name='%s%s_relu' % (name, suffix))
    return act


def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data,
                              num_filter=num_filter,
                              kernel=kernel,
                              num_group=num_group,
                              stride=stride,
                              pad=pad,
                              no_bias=True,
                              name='%s%s_conv2d' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv,
                          name='%s%s_batchnorm' % (name, suffix),
                          fix_gamma=False,
                          momentum=0.9)
    return bn


def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                name='%s%s_conv_sep' % (name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride,
                   name='%s%s_conv_dw' % (name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                  name='%s%s_conv_proj' % (name, suffix))
    return proj


def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity = data
    for i in range(num_block):
        shortcut = identity
        conv = DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group,
                         name='%s%s_block' % (name, suffix), suffix='%d' % i)
        identity = conv + shortcut
    return identity


def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
    body = last_conv
    if fc_type == 'E':
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5, momentum=0.9, name='bn1')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1')
    elif fc_type == "GDC":  # mobilefacenet_v1
        conv_6_dw = Linear(last_conv, num_filter=input_channel, num_group=input_channel, kernel=(7, 7), pad=(0, 0),
                           stride=(1, 1), name="conv_6dw7_7")
        conv_6_f = mx.sym.FullyConnected(data=conv_6_dw, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1')
    return fc1


def get_symbol():
    blocks = [1, 4, 6, 2]
    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125
    conv_1 = Conv(data,
                  num_filter=64,
                  kernel=(3, 3),
                  pad=(1, 1),
                  stride=(2, 2),
                  name="conv_1")
    conv_2_dw = Conv(conv_1,
                     num_group=64,
                     num_filter=64,
                     kernel=(3, 3),
                     pad=(1, 1),
                     stride=(1, 1),
                     name="conv_2_dw")
    conv_23 = DResidual(conv_2_dw,
                        num_out=64,
                        kernel=(3, 3),
                        stride=(2, 2),
                        pad=(1, 1),
                        num_group=128,
                        name="dconv_23")
    conv_3 = Residual(conv_23,
                      num_block=blocks[1],
                      num_out=64,
                      kernel=(3, 3),
                      stride=(1, 1),
                      pad=(1, 1),
                      num_group=128,
                      name="res_3")
    conv_34 = DResidual(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_34")
    conv_4 = Residual(conv_34, num_block=blocks[2], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      num_group=256, name="res_4")
    conv_45 = DResidual(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_45")
    conv_5 = Residual(conv_45, num_block=blocks[3], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      num_group=256, name="res_5")
    conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")
    fc1 = get_fc1(conv_6_sep, 128, 'GDC')
    return fc1


embedding = get_symbol()
gt_label = mx.symbol.Variable('softmax_label')
_weight = mx.symbol.Variable("fc7_weight",
                             shape=(410, 128),
                             lr_mult=1.0,
                             wd_mult=1.0,
                             init=mx.init.Normal(0.01))
s = 64.0
_weight = mx.symbol.L2Normalization(_weight, mode='instance')
nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
fc7 = mx.sym.FullyConnected(data=nembedding,
                            weight=_weight,
                            no_bias=True,
                            num_hidden=410,
                            name='fc7')
zy = mx.sym.pick(fc7, gt_label, axis=1)
cos_t = zy / s
t = mx.sym.arccos(cos_t)
t = t + 0.5
body = mx.sym.cos(t)
new_zy = body * s
diff = new_zy - zy
diff = mx.sym.expand_dims(diff, 1)
gt_one_hot = mx.sym.one_hot(gt_label, depth=410, on_value=1.0, off_value=0.0)
body = mx.sym.broadcast_mul(gt_one_hot, diff)
fc7 = fc7 + body
softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
print(softmax.get_internals())
