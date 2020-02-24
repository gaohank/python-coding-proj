from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import cv2
import mxnet as mx
import numpy as np
import sklearn.preprocessing as pro

from mtcnn_detector import MtcnnDetector

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_preprocess


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # 提取某一层为特征输出层，真正的特征层是交叉熵层前面的高维向量层
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']

    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        self.args = args
        if args.gpu >= 0:
            ctx = mx.gpu(args.gpu)
        else:
            ctx = mx.cpu()
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        if len(args.model) > 0:
            self.model = get_model(ctx, image_size, args.model, 'fc1')

        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        # self.det_factor = 0.9
        self.image_size = image_size
        mtcnn_path = os.path.join(os.path.dirname(__file__), '../models/mtcnn-model')
        if args.det == 0:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=[0.0, 0.0, 0.2])
        self.detector = detector

    def get_input(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[1, 0:4]
        points = points[1, :].reshape((2, 5)).T
        print(bbox)
        print(points)
        # 通过bbox从原图中扣取人脸，并resize为image_size大小
        nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
        # cv2加载的人脸图像默认是BGR，需要转成RGB
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        # 进行维度转化
        aligned = np.transpose(nimg, (2, 0, 1))
        # 维度增加一维，变成(1, 3, 112, 112)
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        return db

    def get_feature(self, db):
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        # 计算每个样本的l2范数，对样本中每个元素除以该范数
        return pro.normalize(embedding).flatten()

    def get_ga(self, data):
        # mxnet模型进行前向推断
        self.model.forward(data, is_train=False)
        ret = self.model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        return gender, age
