import base64

import cv2
import numpy as np

from ocr.service.DecisionBright import decisionBright
from ocr.service.ImagePreProcess import imagePreProcess
from ocr.service.OCRProcess import *

# 이미지 전처리 -> ocr -> 약 찾기( 유사도 분석 )

def ocdDataFormat(text_list):

    data = {
        'hospitalName': findHospital(text_list),
        'phoneNumber': findPhoneNumber(text_list),
        'diseaseCode': findDiseaseCode(text_list),
        'pillNameList': findPillName(text_list),
    }

    print(data)

    return data

def processOCRService(image_data):

    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 이미지 밝기 판단
    image = decisionBright(image);

    # 이미지 전처리
    image = imagePreProcess(image);
    cv2.imwrite('C:/Users/xcxc4/Desktop/pillgood-ocr/ocr/ocr_image/imagePreProcess.png', image)

    # 이미지 텍스트 추출
    text_list = ocrProcess(image)

    # OCR 처리
    result_data = ocdDataFormat(text_list)
    # test

    print(result_data)
    return result_data

