import torch_models
from config import opt
import cv2
import numpy as np
import torch
from mtcnn_detector import MtcnnDetector
import mxnet as mx
import face_preprocess

liveness_model_path = "../models/live.pth"
# 配置模型模式为验证模式
liveness_model = getattr(torch_models, opt.model)().eval()
liveness_model.load(liveness_model_path)

# For dropout, when train(True), it does dropout; when train(False) it doesn’t do dropout (identitical output).
# And for batchnorm, train(True) uses batch mean and batch var; and train(False) use running mean and running var.
liveness_model.train(False)

# 加载mtcnn模型，使用cpu模式
detector = MtcnnDetector(model_folder='../models/mtcnn-model', ctx=mx.cpu(), num_worker=1,
                         accurate_landmark=True, threshold=[0.6, 0.7, 0.8])

# 加载图片
img = cv2.imread("../images/t1.jpg")

bbox, points = detector.detect_face(img, 0)
bbox_size = bbox.shape[0]
points_size = points.shape[0]
for i in range(bbox_size):
    nbbox = bbox[i, 0:4]
    npoints = points[i, :].reshape((2, 5)).T
    img = face_preprocess.preprocess(img, nbbox, npoints, image_size='224,224')

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    data = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    data = data[np.newaxis, :]
    data = torch.FloatTensor(data)
    with torch.no_grad():
        if opt.use_gpu:
            data = data.cuda(1)
        outputs = liveness_model(data)
        outputs = torch.softmax(outputs, dim=-1)
        preds = outputs.to('cpu').numpy()
        attack_prob = preds[:, opt.ATTACK]
        print("attacl_prob:", attack_prob)
        threshold = 0.7
        if attack_prob < threshold:
            print(True)
        else:
            print(False)
