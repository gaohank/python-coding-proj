import cv2
import numpy as np
import random
import copy
import tensorflow as tf
import align.align_dataset_mtcnn
import glob
import os
from scipy import misc
import sys
from PIL import Image,ImageEnhance,ImageOps,ImageFile
import matplotlib.pyplot as plt
import skimage
from skimage import util,io
import tensorflow as tf
import math

def scala_img(img,scale):
    """图片像外或者向内缩放，skimage"""
    return skimage.transform.rescale(img,scale = scale,mode ="constant")

def sp_noise(image,prob):
    '''
    给图片添加椒盐噪声
    :param image: 图片数组
    :param prob:产生亮暗
    :return:
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
          rdn = random.random()
          if rdn < prob:
            output[i][j] = 0
          elif rdn > thres:
            output[i][j] = 255
          else:
            output[i][j] = image[i][j]
    return output

def get_white_num_from_rgb(path):
    """
    输出图片中白点的数量
    :param path: rgb彩色图像路径
    :return: 彩色图片转2值图片的255值（白点）数量
    """
    img = cv2.imread(path)
    Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(Grayimg, 12, 255, cv2.THRESH_BINARY)
    print(thresh.shape)
    num = 0
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            value = thresh[i][j]
            if value == 255:
                num += 1
    return num

def Horizontal_mirroring(image):
    """水平翻转"""
    size = image.shape
    iLR =  copy.deepcopy(image)
    h = size[0]
    w = size[1]
    for i in range(h):
        for j in range(w):
            iLR[i, w - 1 - j] = image[i, j]
    return iLR

def adjust_gamma(image, gamma=1.0):
    """gamma矫正"""


























    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

def main():
    imgdirs = glob.glob("D:\\dingding\\xiazai\\video_photo\\fzh\\*\\*.jpg")
    output_dir = "D:\\dingding\\xiazai\\video_photo\\cp"
    gamma_1 = 0.6
    gamma_2 = 1.7
    print(len(imgdirs))

    for imgdir in imgdirs:
        sess = tf.InteractiveSession()
        img = cv2.imread(imgdir)
        img_name = imgdir.split(os.path.sep)[-1]
        output = os.path.join(output_dir,imgdir.split(os.path.sep)[-2])
        if not os.path.exists(output):
            os.mkdir(output)
        file_name = os.path.join(output,img_name)
        cv2.imwrite(file_name,img)
        print("[INFO] saved {}".format(file_name))
        gamma_img = adjust_gamma(img,gamma_1)
        img_gamma_1_name = img_name.replace(".jpg","_gam{}.jpg".format(gamma_1))
        img_gamma_1_name = os.path.join(output,img_gamma_1_name)
        cv2.imwrite(img_gamma_1_name,gamma_img)
        print("[INFO] saved {}".format(img_gamma_1_name))
        gamma_img = adjust_gamma(img,gamma_2)
        img_gamma_2_name = img_name.replace(".jpg","_gam{}.jpg".format(gamma_2))
        img_gamma_2_name = os.path.join(output,img_gamma_2_name)
        cv2.imwrite(img_gamma_2_name,gamma_img)
        print("[INFO] saved {}".format(img_gamma_2_name))
        gs_noise_img = util.random_noise(img,mode="gaussian")*255 #高斯噪声
        img_gs_name = img_name.replace(".jpg","_gs.jpg")
        img_gs_name = os.path.join(output,img_gs_name)
        cv2.imwrite(img_gs_name,gs_noise_img)
        print("[INFO] saved {}".format(img_gs_name))
        sp_noise_img = sp_noise(img, 0.01)
        img_sp_name = img_name.replace(".jpg", "_sp.jpg")
        img_sp_name = os.path.join(output, img_sp_name)
        cv2.imwrite(img_sp_name,sp_noise_img)
        print("[INFO] saved {}".format(img_sp_name))

        crop_img = tf.random_crop(img,[math.ceil(img.shape[0]*0.8),math.ceil(img.shape[1]*0.8),3]) #随机裁剪
        img_crop_name = img_name.replace(".jpg", "_crop.jpg")
        img_crop_name = os.path.join(output,img_crop_name)
        cv2.imwrite(img_crop_name,crop_img.eval())
        print("[INFO] saved {}".format(img_crop_name))
        hm_img = Horizontal_mirroring(img)
        img_hm_name = img_name.replace(".jpg", "_hm.jpg")
        img_hm_name = os.path.join(output,img_hm_name)
        cv2.imwrite(img_hm_name,hm_img)
        print("[INFO] saved {}".format(img_hm_name))
        # random_contrast_img = tf.image.random_contrast(img,lower=0.2,upper=1.8) #随机对比度
        # img_contrast_name = img_name.replace(".jpg", "_contrast.jpg")
        # img_contrast_name = os.path.join(output,img_contrast_name)
        # cv2.imwrite(img_contrast_name,random_contrast_img.eval())
        # print("[INFO] saved {}".format(img_contrast_name))
        # random_satu_img = tf.image.random_saturation(img, lower=0.2, upper=1.8) #随机饱和度
        # img_satu_name = img_name.replace(".jpg", "_satu.jpg")
        # img_satu_name = os.path.join(output,img_satu_name)
        # cv2.imwrite(img_satu_name,random_satu_img.eval())
        # print("[INFO] saved {}".format(img_satu_name))
        img_re_l = cv2.resize(img, (int(img.shape[1] * 1.5), int(img.shape[0] * 1.5)))
        img_re_l = cv2.resize(img_re_l, (int(img.shape[1]), int(img.shape[0])))
        img_re_l_name = img_name.replace(".jpg", "_re1.5.jpg")
        img_re_l_name = os.path.join(output, img_re_l_name)
        cv2.imwrite(img_re_l_name, img_re_l)
        print("[INFO] saved {}".format(img_re_l_name))
        img_re_s = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
        img_re_s = cv2.resize(img_re_s, (int(img.shape[1]), int(img.shape[0])))
        img_re_s_name = img_name.replace(".jpg", "_re0.5.jpg")
        img_re_s_name = os.path.join(output, img_re_s_name)
        cv2.imwrite(img_re_s_name, img_re_s)
        print("[INFO] saved {}".format(img_re_s_name))
        sess.close()
if __name__ == '__main__':
    main()
    # path = "D:\\dingding\\xiazai\\video_photo\\cp\\GaoZhiQiang_1.jpg"
    # img = cv2.imread(path)
    # print(img.shape)
    # img_m = cv2.resize(img,(int(img.shape[1]*0.7),int(img.shape[0]*0.7)))
    # print(img_m.shape)
    # cv2.imwrite("D:\\dingding\\xiazai\\video_photo\\cp\\2.jpg",img_m)
    # img_m_m = cv2.resize(img_m,(img.shape[1],img.shape[0]))
    # print(img_m_m.shape)
    # cv2.imwrite("D:\\dingding\\xiazai\\video_photo\\cp\\3.jpg",img_m_m)



