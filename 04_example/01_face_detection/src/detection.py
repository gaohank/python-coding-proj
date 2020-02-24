from retinaface import RetinaFace
import numpy as np
import cv2


def get_scale(img_shape):
    scales = [1024, 1980]
    im_shape = img_shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]

    return scales


# 读取图片
img = cv2.imread('../images/t2.jpg')

detector = RetinaFace("../models/mnet.25", 0, -1, 'net3')

threshold = 0.8

scale = get_scale(img.shape)

bbox_1, landmarks_1 = detector.detect(img, threshold, scale)

print(bbox_1)
print(landmarks_1)

faces = bbox_1.tolist()

for i in range(len(faces)):
    img = cv2.rectangle(img, (int(faces[i][0]), int(faces[i][1])), (int(faces[i][2]), int(faces[i][3])), (0, 0, 255), 2, 8, 0)

cv2.imshow("img", img)
cv2.waitKey(0)