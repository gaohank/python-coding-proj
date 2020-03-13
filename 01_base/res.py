import os

import cv2


def get_file_list(dir_path, new_file_list):
    if os.path.isfile(dir_path):
        new_file_list.append(dir_path)
    elif os.path.isdir(dir_path):
        for s in os.listdir(dir_path):
            new_dir = os.path.join(dir_path, s)
            get_file_list(new_dir, new_file_list)
    return new_file_list


report_dir = r'D:\tct\792\2019'
report_list = get_file_list(report_dir, [])

for i in range(len(report_list)):
    img = cv2.imread(report_list[i])
    sp = img.shape
    print(sp)
    if sp[0] == 1962:
        cv2.imwrite('D:\\tct\\792\\report' + '\\' + str(i + 1) + '.Jpeg', img)
    else:
        cv2.imwrite('D:\\tct\\792\\img' + '\\' + str(i + 1) + '.Jpeg', img)

if __name__ == "__main__":
    print("123")
