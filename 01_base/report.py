import os

import pytesseract
from PIL import Image


# 文件夹递归获取图片路径
def get_file_list(dir_path, new_file_list):
    if os.path.isfile(dir_path):
        new_file_list.append(dir_path)
    elif os.path.isdir(dir_path):
        for s in os.listdir(dir_path):
            new_dir = os.path.join(dir_path, s)
            get_file_list(new_dir, new_file_list)
    return new_file_list


report = r'D:\tct\792\report_same'  # report_same
img = r'D:\tct\792\img_same'  # img_same
report_list = get_file_list(report, [])
img_list = get_file_list(img, [])
for i in range(len(report_list)):
    print(report_list[i])
    report_filename = os.path.basename(report_list[i])

    img = Image.open(report_list[i])
    region = (65, 1320, 700, 1430)
    cropImg = img.crop(region)
    # cropImg.show()
    result = pytesseract.image_to_string(cropImg, lang='chi_sim')

    if '细胞不能明确意义' in result or 'ASC--US' in result or 'ASC-US' in result:
        # print('细胞不能明确意义')
        for j in range(len(img_list)):
            img_filename = os.path.basename(img_list[j])
            if report_filename == img_filename:
                img_bad = Image.open(img_list[j])
                save_img = img_bad.save('D:\\tct\\792\\class\\ASC-US\\' + img_filename)  # 放到class下得ASC-US文件夹中
                break
    elif '细胞倾向于上皮内病变' in result or 'ASC-H' in result:
        # print('细胞倾向于上皮内病变')
        for j in range(len(img_list)):
            img_filename = os.path.basename(img_list[j])
            if report_filename == img_filename:
                img_bad = Image.open(img_list[j])
                save_img = img_bad.save('D:\\tct\\792\\class\\ASC-H\\' + img_filename)  # 放到class下得ASC-H文件夹中
                break

    elif '上皮内低度病变' in result or 'LSIL' in result or 'LSTL' in result or 'LSIL' in result:
        # print('上皮内低度病变')
        for j in range(len(img_list)):
            img_filename = os.path.basename(img_list[j])
            if report_filename == img_filename:
                img_bad = Image.open(img_list[j])
                save_img = img_bad.save('D:\\tct\\792\\class\\LSIL\\' + img_filename)  # 放到class下得LSIL文件夹中
                break

    elif '上皮内高度病变' in result or 'HSIL' in result or 'HSTL' in result:
        # print('上皮内高度病变')
        for j in range(len(img_list)):
            img_filename = os.path.basename(img_list[j])
            if report_filename == img_filename:
                img_bad = Image.open(img_list[j])
                save_img = img_bad.save('D:\\tct\\792\\class\\HSIL\\' + img_filename)  # 放到class下得HSIL文件夹中
                break

    elif '未见上皮内病变及恶性细胞' in result or 'NTLM' in result or 'NILM' in result:
        # print('未见上皮内病变及恶性细胞')
        for j in range(len(img_list)):
            img_filename = os.path.basename(img_list[j])
            if report_filename == img_filename:
                print('111')
                img_bad = Image.open(img_list[j])
                save_img = img_bad.save('D:\\tct\\792\\class\\NILM\\' + img_filename)  # 放到class下得NILM文件夹中
                break

    else:
        # print('其他病变')
        for j in range(len(img_list)):
            img_filename = os.path.basename(img_list[j])

            if report_filename == img_filename:
                img_bad = Image.open(img_list[j])
                save_img = img_bad.save('D:\\tct\\792\\class\\other\\' + img_filename)  # 放到class下得other文件夹中
                break
