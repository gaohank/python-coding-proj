import requests
import json
from base64 import b64encode

ENCODING = "utf-8"
path_1 = 'idcard.jpg'
path_2 = 'gsl.jpg'
with open(path_1, 'rb') as img:
    byte_content = img.read()
    base64_bytes = b64encode(byte_content)
    base64_string = base64_bytes.decode(ENCODING)
    print(base64_string)
with open(path_2, 'rb') as img2:
    byte_content_2 = img2.read()
    base64_bytes_2 = b64encode(byte_content_2)
    base64_string_2 = base64_bytes_2.decode(ENCODING)
    print(base64_string_2)

raw_data = dict()
raw_data["IdCardNo"] = "876571186"
raw_data["Photo"] = base64_string
raw_data["IdCardPic"] = base64_string_2

print(json.dumps(raw_data))
# _ = requests.post('http://10.18.18.196:7700/faceRecognition_Server', data=json.dumps(raw_data))
_ = requests.post('http://staging.facerecognition.aio.ai.ikang.com/faceRecognition_Server', data=json.dumps(raw_data))
print(_.text)

