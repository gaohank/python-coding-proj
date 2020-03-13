import os
from PIL import Image
from numpy import average, dot, linalg


# 对图片进行统一化处理
def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res


def get_file_list(dir_path, new_file_list):
    if os.path.isfile(dir_path):
        new_file_list.append(dir_path)
    elif os.path.isdir(dir_path):
        for s in os.listdir(dir_path):
            new_dir = os.path.join(dir_path, s)
            get_file_list(new_dir, new_file_list)
    return new_file_list

report_dir = r'D:\tct\122\report'
img_dir = r'D:\tct\122\img'
report_list = get_file_list(report_dir, [])
img_list = get_file_list(img_dir, [])

for i in range(len(report_list)):
    report = Image.open(report_list[i])
    region = (748, 1185, 1235, 1550)
    report_crop = report.crop(region)
    for j in range(len(img_list)):
        image = Image.open(img_list[j])
        img = image.resize((482, 360), Image.ANTIALIAS)
        cosin = image_similarity_vectors_via_numpy(report_crop, img)
        # print(cosin)
        if cosin >= 0.995 :
            # report_same = report.save("F:\\data\\TCT\\test\\report_same" + '\\' + str(i+1) + '.Jpeg')
            report_same = report.save("D:\\tct\\122\\report_same" + '\\' + str(i+1) + '.Jpeg')
            # img_same = img.save("F:\\data\\TCT\\test\\img_same" + '\\' + str(i+1) + '.Jpeg')
            img_same = image.save("D:\\tct\\122\\img_same" + '\\' + str(i+1)+ '.Jpeg')
            break
        else:
            continue
    print('还需等待' + str(len(report_list) - (i+1))+'图片')
