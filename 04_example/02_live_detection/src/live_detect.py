import torch_models
from config import opt
import cv2
import numpy as np
import torch

liveness_model_path = "../models/live.pth"
# 配置模型模式为验证模式
liveness_model = getattr(torch_models, opt.model)().eval()
liveness_model.load(liveness_model_path)

# For dropout, when train(True), it does dropout; when train(False) it doesn’t do dropout (identitical output).
# And for batchnorm, train(True) uses batch mean and batch var; and train(False) use running mean and running var.
liveness_model.train(False)

# 加载图片
img = cv2.imread("../images/t2.jpg")
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
