import base64

import cv2
import numpy as np

# 이미지 밝기 판단
def calculate_brightness(image):

    # 이미지를 NumPy 배열로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness

def adjust_brightness(image, brightness):

    # 조정된 이미지를 위한 빈 행렬 생성
    adjusted_image = np.empty_like(image)

    # cv2.add 연산을 사용하여 밝기 조정
    cv2.convertScaleAbs(image, adjusted_image, 1, brightness)

    return adjusted_image

def decisionBright(image):
    print("이미지 밝기 판단 실행")

    bright = calculate_brightness(image)

    print(bright)

    #기본적인 밝기 처리
    if(bright > 170):
        brightness = -30
        adjusted_image = adjust_brightness(image, brightness)
        cv2.imwrite('C:/Users/xcxc4/Desktop/pillgood-ocr/ocr/ocr_image/adjusted_image.png', adjusted_image)

        return adjusted_image
    elif(bright < 130):
        brightness = 30
        adjusted_image = adjust_brightness(image, brightness)
        cv2.imwrite('C:/Users/xcxc4/Desktop/pillgood-ocr/ocr/ocr_image/adjusted_image.png', adjusted_image)

        return adjusted_image

    return image