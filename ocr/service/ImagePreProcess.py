import cv2
import numpy as np

# 이미지 전처리부분
def colorToBlackBinary(image):
    brightness = 0  # 조정할 밝기 값
    adjusted_image = np.uint8(np.clip(image.astype(np.int32) + brightness, 0, 255))

    # 그레이스케일 변환
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

    # 적응적 이진화
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 30
    )

    # 구조화 요소 (커널) 생성
    kernel = np.ones((2, 2), np.uint8)

    # 팽창 모폴로지 연산 수행
    dilate_image = cv2.dilate(binary_image, kernel)

    # 블러 처리
    blurred_image = cv2.GaussianBlur(dilate_image, (3, 3), 0)

    return blurred_image

def colorToWhiteBinary(image):
    brightness = 0  # 조정할 밝기 값
    adjusted_image = np.uint8(np.clip(image.astype(np.int32) + brightness, 0, 255))

    # 그레이스케일 변환
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

    # 적응적 이진화
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 501, 30
    )

    # 블러 처리
    blurred_image = cv2.GaussianBlur(binary_image, (3, 3), 0)

    return blurred_image


def croppedImage(dilate_image, image):
    ### selecting min size as 15 pixels
    line_min_width = 15
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)

    # Horizontal Kernel, Vertical Kernel
    img_bin_h = cv2.morphologyEx(dilate_image, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(dilate_image, cv2.MORPH_OPEN, kernal_v)

    # MIX Kernel
    img_bin_final = img_bin_h | img_bin_v

    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for x, y, w, h, area in stats[2:]:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x + w > max_x:
                max_x = x + w
            if y + h > max_y:
                max_y = y + h

    cropped_image = image[min_y:max_y, min_x:max_x]

    return cropped_image

def imageProcess(image):

    dilate_image = colorToBlackBinary(image)
    cv2.imwrite('C:/Users/xcxc4/Desktop/pillgood-ocr/ocr/ocr_image/dilate_image.png', dilate_image)

    cropImage = croppedImage(dilate_image, image)
    cv2.imwrite('C:/Users/xcxc4/Desktop/pillgood-ocr/ocr/ocr_image/cropImage.png', cropImage)

    finalImage = colorToWhiteBinary(cropImage)
    cv2.imwrite('C:/Users/xcxc4/Desktop/pillgood-ocr/ocr/ocr_image/finalImage.png', finalImage)

    return finalImage

# Press the green button in the gutter to run the script.
def imagePreProcess(photo):
    print("이미지 전처리 실행")
    finalImage = imageProcess(photo)

    return finalImage