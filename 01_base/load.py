import numpy as np
import cv2 as cv
import io
from PIL import Image

# image_stream = io.BytesIO()
# image_stream.write(connection.read(image_len))
# image_stream.seek(0)
# file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
# img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

img = cv.imread("5.jpg")
cv.imshow("src", img)
cv.waitKey(0)

import base64


# 图片转字节
def tu_zi_jie():
    with open('5.jpg', 'rb') as fp:
        tu = base64.b64encode(fp.read())
    # 生成很长得一串字节流
    print(tu)
    return tu


# 字节转图片
def zi_tu(tu):
    b = base64.b64decode(tu)
    print(b)
    # with open('tu.png', 'wb') as fp:
    #     fp.write(b)


# if __name__ == '__main__':
#     t = tu_zi_jie()
#     zi_tu(t)
