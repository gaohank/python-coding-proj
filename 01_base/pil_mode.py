from PIL import Image
import numpy as np
import os

oridir = r"C:\Users\Administrator\Desktop\dxImageAnnotation\JPEGImages"
listdir = os.listdir(oridir)
print(listdir.pop(1))
print(len(os.listdir(oridir)))

# dir = "D:/02-code/rvos/databases/DAVIS2017/Annotations/480p/"
# dirs = os.listdir(dir)
#
# for d in dirs:
#     rootpath = os.path.join(dir, d)
#     size = len(os.listdir(rootpath))
#     for i in range(size):
#         path = os.path.join(rootpath + '/', '%05d.png' % i)
#         img = Image.open(path)
#         if img.mode == 'F':
#             print(d, i)

# for i in range(82):
#     path = os.path.join('D:/02-code/rvos/databases/DAVIS2017/JPEGImages/480p/bear/', '%05d.jpg' % i)
#     img = Image.open(path)
#     print(img.mode == 'F')
