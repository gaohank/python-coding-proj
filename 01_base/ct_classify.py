import os
import shutil
import cv2
import numpy as np
import pytesseract

image_input_dir = r'E:\3-资料\3-业务文档\6-超声诊断\会员822\会员822'
image_output_path = r'E:\3-资料\3-业务文档\6-超声诊断\会员822\会员822'


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
    # return arr[len(arr) - 1].replace(".","")


# 识别部位并分类
def ocr_thy(img):
    # 读取图片
    img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(img)
    return text


if __name__ == '__main__':
    file_list = get_file_list(image_input_dir, [])
    for each_file in file_list:
        if get_file_extension(each_file) == '.jpg':
            # 输入一张图片的路径，返回分类，定义为class_name
            print(each_file)
            class_name = ocr_thy(each_file)
            if 'Thyroid' in class_name:
                img_path = image_output_path + '\\Thyroid' + '\\'
                print(img_path)
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Vasc Carotid' in class_name:
                img_path = image_output_path + '\\Vasc Carotid' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Adult Echo' in class_name:
                img_path = image_output_path + '\\Adult Echo' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Abd Gen' in class_name:
                img_path = image_output_path + '\\Abd Gen' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Breast' in class_name:
                img_path = image_output_path + '\\Breast' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Gyn Pelvis' in class_name:
                img_path = image_output_path + '\\Gyn Pelvis' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Vasc Penl' in class_name:
                img_path = image_output_path + '\\Vasc Penl' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Abdomen' in class_name:
                img_path = image_output_path + '\\Abdomen' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'LEV' in class_name:
                img_path = image_output_path + '\\LEV' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Gyn' in class_name:
                img_path = image_output_path + '\\Gyn' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Adult' in class_name:
                img_path = image_output_path + '\\Adult' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Carotid' in class_name:
                img_path = image_output_path + '\\Carotid' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
            if 'Adv Breast' in class_name:
                img_path = image_output_path + '\\Adv Breast' + '\\'
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                shutil.copy(each_file, img_path)
    print("文件夹中图片总张数为：" + str(len(file_list)))
