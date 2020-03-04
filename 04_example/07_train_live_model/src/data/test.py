import glob
import os
import random

import numpy as np

# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
imgdirs = glob.glob("E:\\cp\\face_recognition\\train\\*\\*\\*.jpg")
file_dir = 'F:\\liveness_dataset\\train'
shuffle =True
file_name_real ="E:\\cp\\face_recognition\\train\\client_train_raw.txt"
file_name_fake ="E:\\cp\\face_recognition\\train\\ imposter_train_raw.txt"
file_name_real_test = "E:\\cp\\face_recognition\\train\\client_test_raw.txt"
file_name_fake_test = "E:\\cp\\face_recognition\\train\\imposter_test_raw.txt"
name = None
data = dict()
for imgdir in imgdirs:
    name = imgdir.split(os.path.sep)[-2]
    if (name in data.keys()):
        data["{}".format(name)].append(imgdir)
    else:
        data["{}".format(name)] = []
        data["{}".format(name)].append(imgdir)
imgdirs_ = []
for key in data :
    data_list = data["{}".format(key)]
    np.random.seed(100)
    np.random.shuffle(data_list)
    for imgdir in data_list:
        imgdirs_.append(imgdir)
print(len(imgdirs_))

with open(file_name_real,'a') as real:
    with open(file_name_fake,'a') as fake:
        with open(file_name_real_test,"a") as real_test:
            with open(file_name_fake_test,"a") as fake_test:
        # if shuffle is True:
        #     np.random.seed(100)
        #     np.random.shuffle(imgdirs)
                for img in imgdirs_:
                    # new_name = img.replace(".png",".jpg")
                    # os.rename(img,new_name)
                    file = img.split(os.path.sep)[-3:]
                    file_ = os.path.join(file[1], file[2])
                    if file[0].endswith("ImposterRaw"):
                        if random.randint(1,10) > 2:
                            fake.write(file_+"\n")
                        else:
                            fake_test.write(file_+"\n")
                    else:
                        if random.randint(1,10) > 2 :
                            real.write(file_+"\n")
                        else:
                            real_test.write(file_+"\n")




