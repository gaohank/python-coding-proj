import os
import cv2
import numpy as np
import pytesseract


# 文件夹递归获取图片路径
def get_file_list(dir_path, new_file_list):
    if os.path.isfile(dir_path):
        new_file_list.append(dir_path)
    elif os.path.isdir(dir_path):
        for s in os.listdir(dir_path):
            new_dir = os.path.join(dir_path, s)
            get_file_list(new_dir, new_file_list)
    return new_file_list


# 后缀名
def get_file_extension(filename):
    arr = os.path.splitext(filename)
    return arr[len(arr) - 1]


def ocr_thy(img):
    # 读取图片
    img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(img)
    return text


if __name__ == '__main__':
    # file_list = get_file_list(r'E:\3-资料\3-业务文档\6-超声诊断\会员822\会员822', [])
    # print(file_list[0])
    # print("文件夹中图片总张数为：" + str(len(file_list)))
    # print(get_file_extension(file_list[0]))
    # print("Healthcare" in ocr_thy(file_list[0]))
    img = cv2.imdecode(np.fromfile(r"E:\3-资料\3-业务文档\6-超声诊断\会员822\会员822\2018\20181210\2018121000001\1.jpg", np.uint8), cv2.IMREAD_COLOR)
    print("Healthcare" in pytesseract.image_to_string(img))
    height, width = img.shape[:2]
    newImg = cv2.resize(img, (int(width*0.3), int(height*0.5)))
    # img.shape[0]得到的是图片的高，img.shape[1]得到是图片的宽
    geTwoImg = img[35:60, 1145:1235]
    # resize 为宽，高
    geTwoOutImage = cv2.resize(geTwoImg, (70, 45))
    # 打印图片
    # cv2.imshow("src", newImg)
    # cv2.imshow("src img", img)

    cv2.imshow("img", img)
    cv2.imshow("geTwoImg", geTwoImg)
    cv2.imshow("geTwoOutImage", geTwoOutImage)

    cv2.imwrite('5.jpg', geTwoOutImage)
    geTwoText = pytesseract.image_to_string(cv2.imread('5.jpg'), "chi_sim")
    print(geTwoText)

    cv2.waitKey(0)
    # print(img)
    # # print(img.shape[0])
    # # print(img.shape[1])
    # geTwoImg = img[5:30, 380:420]
    # geTwoOutImage = cv2.resize(geTwoImg, (70, 45))
    # cv2.imwrite('5.jpg', geTwoOutImage)
    # geTwoText = pytesseract.image_to_string(cv2.imread('5.jpg'), lang='chi_sim')
    # print(geTwoText)
