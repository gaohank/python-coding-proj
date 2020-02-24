import cv2

# 默认BGR格式加载
img = cv2.imread("images/1.jpg")

# 转为灰度图像，占用空间更小
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', gray)
cv2.waitKey(0)
