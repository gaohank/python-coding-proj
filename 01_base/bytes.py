import io
from PIL import Image
import numpy as np
import cv2
import mxnet

img = Image.open("5.jpg")
print(img)
image_bytes = open('5.jpg', 'rb').read()
print(image_bytes)
print(len(image_bytes))
image_nparray = np.frombuffer(image_bytes, np.uint8)
# print(image_nparray)
# print(image_nparray[2183])
# print(image_nparray[2184])
# print(image_nparray[2185])
# print(len(image_nparray))
image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
print(image.shape)
print(image.shape[0:2])
