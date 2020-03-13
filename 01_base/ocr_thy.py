from PIL import Image
import pytesseract
import cv2
import numpy as np


def ocr_thy(img):
    # 读取图片
    img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    # img = cv2.imread(img)
    imgshape = img.shape[1]
    if imgshape == 1920:
        geImage = img[5:30, 685:730]
        geOutImage = cv2.resize(geImage, (70, 45))
        epiqImage = img[15:35, 1090:1140]
        epiqOutImage = cv2.resize(epiqImage, (90, 40))

        cv2.imwrite('1.jpg', geOutImage)
        cv2.imwrite('2.jpg', epiqOutImage)

        geText = pytesseract.image_to_string(cv2.imread('1.jpg'), lang='chi_sim')

        epiqText = pytesseract.image_to_string(cv2.imread('2.jpg'), lang='chi_sim')

        if geText == 'GE' and epiqText == '':
            geThyImage = img[35:65, 1490:1580]
            geThyOutImage = cv2.resize(geThyImage, (80, 45))
            cv2.imwrite('3.jpg', geThyOutImage)
            text = pytesseract.image_to_string(cv2.imread('3.jpg'), lang='chi_sim')
            return text

        if geText == '' and epiqText == 'EPIQ5':
            epiqThyImage = img[130:160, 500:600]
            epiqThyOutImage = cv2.resize(epiqThyImage, (80, 35))
            cv2.imwrite('4.jpg', epiqThyOutImage)
            text = pytesseract.image_to_string(cv2.imread('4.jpg'), lang='chi_sim')
            return text

    if imgshape == 1280:
        geTwoImg = img[5:30, 380:420]
        geTwoOutImage = cv2.resize(geTwoImg, (70, 45))
        cv2.imwrite('5.jpg', geTwoOutImage)
        geTwoText = pytesseract.image_to_string(cv2.imread('5.jpg'), lang='chi_sim')

        if geTwoText == 'GE':
            geTwoThyImage = img[35:60, 1145:1235]
            geTwoThyOutImage = cv2.resize(geTwoThyImage, (80, 40))
            cv2.imwrite('6.jpg', geTwoThyOutImage)
            text = pytesseract.image_to_string(cv2.imread('6.jpg'), lang='chi_sim')
            return text
