import os
import shutil


def get_file_list(dir_path, new_file_list):
    if os.path.isfile(dir_path):
        new_file_list.append(dir_path)
    elif os.path.isdir(dir_path):
        for s in os.listdir(dir_path):
            new_dir = os.path.join(dir_path, s)
            if os.path.isfile(new_dir):
                get_file_list(os.path.join(dir_path, os.listdir(dir_path)[-1]), new_file_list)
                return new_file_list
            get_file_list(new_dir, new_file_list)
    return new_file_list


pic_dir = r'D:\01-工作文档\02-项目\13-考勤\train_data'
pic_list = get_file_list(pic_dir, [])

print(pic_list)
if not os.path.exists("./test"):
    os.mkdir("test")

for pic in pic_list:
    shutil.copy(pic, "test/" + os.path.basename(pic))
