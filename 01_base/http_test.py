import cv2
import numpy as np
import tensorflow as tf
import base64

img = cv2.resize(cv2.imread(r"D:\CTData\JPEGImages\0.jpg"), (416, 416))
# base = base64.b64encode(img)
# s = str(base, encoding='utf-8').replace('+', '-').replace(r'/', '_')
# print(s)
# print(base)
array = np.array([img])
# print(array.shape)

url = 'http://localhost:8501/v1/models/Thyroid:predict'

tensor = tf.convert_to_tensor(array)
# print(array.tostring())
np.set_printoptions(threshold=np.inf)

predict_request = '{"instances": %s}' % str(base64.b64encode(img), encoding='utf-8').replace('+', '-').replace(r'/', '_')  # 一定要list才能传输，不然json错误
# print(tensor)
replace = predict_request.replace("array(", "").replace(", dtype=uint8)", "").replace("\n", "").replace(" ", "")
# print(replace)


# print("start")
# start_time = time()
# r = requests.get(url, data=predict_request)
# print(r.content)
# end_time = time()
