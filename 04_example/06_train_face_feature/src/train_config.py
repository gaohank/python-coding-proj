import numpy as np
import os
from easydict import EasyDict as ed

config = ed()

config.bn_mom = 0.9  # momentum
config.workspace = 256  # mxnet需要的缓冲空间
config.emb_size = 512  # 输出的特征数量的维度
config.ckpt_embedding = True  # 是否检测输出的特征向量
config.net_se = 1  # 网络结构中添加se模块
config.net_act = 'prelu'  # 激活函数
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1, 4, 6, 2]
config.net_output = 'E'  # 输出层，链接层的类型，如"GDC"也是其中一种，具体查看recognition\symbol\symbol_utils.py
config.net_multiplier = 1.0
# config.val_targets = ['lfw', 'cfp_fp', 'agedb_30', 'calfw', 'cfp_ff']  # 测试数据，即.bin为后缀的文件
config.ce_loss = True  # Focal loss
config.fc7_lr_mult = 1.0  # 学习率的倍数
config.fc7_wd_mult = 1.0  # 权重刷衰减的倍数
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = True  # 数据随机进行镜像翻转
config.data_cutoff = False  # 数据进行随机裁剪
config.data_color = 0  # 数据进行彩色增强
config.data_images_filter = 0
config.count_flops = True  # 是否计算一个网络占用的浮点数内存
config.memonger = False  # not work now

# network settings
# r50 r50v1 d169 d201 y1 m1 m05 mnas mnas025
network = ed()

network.r100 = ed()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r100fc = ed()
network.r100fc.net_name = 'fresnet'
network.r100fc.num_layers = 100
network.r100fc.net_output = 'FC'

network.r50 = ed()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50v1 = ed()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1

network.d169 = ed()
network.d169.net_name = 'fdensenet'
network.d169.num_layers = 169
network.d169.per_batch_size = 64
network.d169.densenet_dropout = 0.0

network.d201 = ed()
network.d201.net_name = 'fdensenet'
network.d201.num_layers = 201
network.d201.per_batch_size = 64
network.d201.densenet_dropout = 0.0

network.y1 = ed()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'

network.y2 = ed()
network.y2.net_name = 'fmobilefacenet'
network.y2.emb_size = 256
network.y2.net_output = 'GDC'
network.y2.net_blocks = [2, 8, 16, 4]

network.m1 = ed()
network.m1.net_name = 'fmobilenet'
network.m1.emb_size = 256
network.m1.net_output = 'GDC'
network.m1.net_multiplier = 1.0

network.m05 = ed()
network.m05.net_name = 'fmobilenet'
network.m05.emb_size = 256
network.m05.net_output = 'GDC'
network.m05.net_multiplier = 0.5

network.mnas = ed()
network.mnas.net_name = 'fmnasnet'
network.mnas.emb_size = 256
network.mnas.net_output = 'GDC'
network.mnas.net_multiplier = 1.0

network.mnas05 = ed()
network.mnas05.net_name = 'fmnasnet'
network.mnas05.emb_size = 256
network.mnas05.net_output = 'GDC'
network.mnas05.net_multiplier = 0.5

network.mnas025 = ed()
network.mnas025.net_name = 'fmnasnet'
network.mnas025.emb_size = 256
network.mnas025.net_output = 'GDC'
network.mnas025.net_multiplier = 0.25

network.vargfacenet = ed()
network.vargfacenet.net_name = 'vargfacenet'
network.vargfacenet.net_multiplier = 1.25
network.vargfacenet.emb_size = 512
network.vargfacenet.net_output = 'J'

# dataset settings
dataset = ed()

dataset.webface = ed()
dataset.webface.dataset = 'webface'
dataset.webface.dataset_path = '/home/amax/iKang/cp/data/face_recognition/webface/merge_file'
dataset.webface.num_classes = 10572
dataset.webface.image_shape = (112, 112, 3)
dataset.webface.val_targets = ['lfw', 'cfp_fp', 'agedb_30', 'calfw', 'cfp_ff']

dataset.retina = ed()
dataset.retina.dataset = 'retina'
dataset.retina.dataset_path = '../datasets/ms1m-retinaface-t1'
dataset.retina.num_classes = 93431
dataset.retina.image_shape = (112, 112, 3)
dataset.retina.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vgg = ed()
dataset.vgg.dataset = "vgg"
dataset.vgg.dataset_path = "/home/amax/iKang/cp/data/face_recognition/vgg/merge_file/"
dataset.vgg.num_classes = 8631
dataset.vgg.image_shape = (112, 112, 3)
dataset.vgg.val_targets = [""]

dataset.ms1m = ed()
dataset.ms1m.dataset = "ms1m"
dataset.ms1m.dataset_path = "/data0/face_recognition/ms1m/merge_file"
dataset.ms1m.num_classes = 85164
dataset.ms1m.image_shape = [112, 112, 3]
dataset.ms1m.val_targets = [""]

dataset.glintasia = ed()
dataset.glintasia.dataset = "glintasia"
dataset.glintasia.dataset_path = "/data0/face_recognition/glintasia/merge_file"
dataset.glintasia.num_classes = 93979
dataset.glintasia.image_shape = [112, 112, 3]
dataset.glintasia.val_targets = [""]

dataset.umd = ed()
dataset.umd.dataset = 'umd'
dataset.umd.dataset_path = "/data0/face_recognition/umd"
dataset.umd.num_classes = 8277
dataset.umd.image_shape = [112, 112, 3]
dataset.umd.val_targets = [""]

# 损失函数
# loss_m1，loss_m2，loss_m3，其出现3个m，作者是为了减少代码量，把多个损失函数合并在一起了
# 即nsoftmax，arcface，cosface，combined
loss = ed()
loss.softmax = ed()
loss.softmax.loss_name = 'softmax'

loss.nsoftmax = ed()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = ed()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.cosface = ed()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = ed()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.triplet = ed()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 5
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 7200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 60
loss.triplet.lr = 0.05

loss.atriplet = ed()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35
loss.atriplet.triplet_bag_size = 7200
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

# default settings
default = ed()

# default network
default.network = 'r100'
default.pretrained = 'pretrain/model-r100-ii/model'
default.pretrained_epoch = 0
# default dataset
default.dataset = ["webface", 'vgg', 'ms1m', 'glintasia', 'umd']
default.loss = 'arcface'
default.frequent = 20  # 每20个批次打印一次准确率等log
default.verbose = 2000  # 每训练2000次，对验证数据进行一次评估
default.kvstore = 'device'  # 键值存储

default.end_epoch = 10000  # 结束的epoch
default.lr = 0.01  # 初始学习率，如果每个批次训练的数目小，学习率也相应的降低
default.wd = 0.0005  # 权重衰减
default.mom = 0.9
default.per_batch_size = 32  # 每存在一个GPU，训练128个批次，如两个GPU，则实际训练的batch_size为128*2
default.ckpt = 1
default.lr_steps = '100000,160000,220000'  # 每达到步数，学习率变为原来的百分之十
default.models_root = 'models'  # 模型保存的位置


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
        config[k] = v
        if k in default:
            default[k] = v
    for k, v in network[_network].items():
        config[k] = v
        if k in default:
            default[k] = v
    if isinstance(_dataset, list):
        for _dataset_seq in _dataset:
            print(dataset[_dataset_seq].items())
            for k, v in dataset[_dataset_seq].items():
                if k not in config.keys():
                    config[k] = list()
                    config[k].append(v)
                else:
                    config[k].append(v)
                if k in default and k != "dataset":
                    if k not in default.keys():
                        default[k] = list()
                        default[k].append(v)
                    else:
                        default[k].append(v)
    else:
        for k, v in dataset[_dataset].items():
            config[k] = v
            if k in default:
                default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
        config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

# generate_config(default.network, default.dataset, default.loss)
