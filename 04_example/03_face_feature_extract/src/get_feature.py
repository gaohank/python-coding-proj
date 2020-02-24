import argparse

import cv2

import face_model
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
# 这个比较重要，人脸识别的model，我的经验是要用绝对路径
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
# 这个比较重要，年龄性别的model，我的经验是要用绝对路径
parser.add_argument('--ga-model', default='../models/gamodel-r50/model,0', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
print(args)
# 加载model
model = face_model.FaceModel(args)
# 读取图片
img = cv2.imread('../images/t2.jpg')
# 模型加载图片
img = model.get_input(img)
# 获得特征
f1 = model.get_feature(img)
# 输出特征
print(f1)
print(np.shape(f1))
