import easyocr
import re

from ocr.service.OCRPostProcess import ocrPostProcess


# ocr처리부분
def findPhoneNumber(text_list):

    # 000-0000에 맞는 정규 표현식 패턴
    pattern = r"\(\d*\)\d{3}-\d{4}"

    # 정규 표현식을 사용하여 요소 필터링
    for text in text_list:
        if re.match(pattern, text):
            print("병원 전화번호 추출 : {}".format(text))
            return text

    # 결과 출력
    return ''


# 레벤슈타인 거리 계산 ( 질분분리기호 위치 찾는 용도 )
def levenshtein_distance(a, b):
    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(
                    matrix[i - 1][j],
                    matrix[i][j - 1],
                    matrix[i - 1][j - 1]
                ) + 1

    return matrix[len(a)][len(b)]


# 레벤슈타인 거리 계산 사용( 질분분리기호 위치 찾는 용도 )
def find_best_match(datalist, search_word, similarity_threshold=50):
    search_word = search_word.lower()
    best_match_idx = -1
    current_best_similarity = -1

    for idx, string in enumerate(datalist):
        edit_distance = levenshtein_distance(string, search_word)
        max_len = max(len(string), len(search_word))
        similarity_percentage = (1 - edit_distance / max_len) * 100
        if similarity_percentage > current_best_similarity and similarity_percentage >= similarity_threshold:
            best_match_idx = idx
            current_best_similarity = similarity_percentage

    return best_match_idx


def get_sublist(lst, index):
    start_index = max(0, index - 5)
    end_index = index + 6  # Add 1 to include the element at index 15

    sublist = lst[start_index:end_index]
    return sublist


def findDiseaseCode(text_list):
    search_word = "질병분류기호"
    search_word_index = find_best_match(text_list, search_word)

    disease_word_list = get_sublist(text_list, search_word_index)
    # A01부터 Z99까지에 해당하는 요소를 찾기 위한 정규 표현식 패턴
    pattern = r"[A-Z][0-9]{2}$"

    # 정규 표현식을 사용하여 요소 필터링 및 반환
    for text in disease_word_list:
        if re.match(pattern, text):
            print("질병분류기호 추출 : {}".format(text))
            return text
    # 결과 없음
    return None


def findHospital(text_list):

    # "병원"으로 끝나는 요소를 찾기 위한 정규 표현식 패턴
    pattern = r".+병원$"

    # 정규 표현식을 사용하여 요소 필터링

    for text in text_list:
        if re.match(pattern, text):
            print("병원 이름 추출 : {}".format(text))
            return text

    return ''
    # 결과 출력


def contains_english(text):
    for char in text:
        if char.isalpha() and char.isascii():
            if char.isupper() or char.islower():
                return True

    return False

def findTakePillTime(takeCount):
    if takeCount == '1':
        return ['2']

    if takeCount == '2':
        return ['2','4']

    if takeCount == '3':
        return ['2', '3', '4']

    if takeCount == '4':
        return ['1', '2', '3', '4']

    if takeCount == '5':
        return ['1', '2', '3', '4', '5']

    return ['']

def is_numeric_string(string):
    return string.isdigit()

# 리스트에서 약 판별
def findPillName(text_list):

    print("약 판별 시작")

    # 약 찾기 위한 유사도 분석
    find_fill = ocrPostProcess(text_list)
    print("찾은 약 : ", find_fill)

    result = []
    for index, item in enumerate(find_fill):
        isTakeCount = False
        target_index = item['index']
        takeCount = 1
        takeDay = 1

        print(text_list[target_index:target_index+4])

        # text_list에 약 이 발견되면 숫자(안나올 수 있음) 영어 target1 target2에서 tartget 2개를 가져와야 됨
        for i in range(target_index+1, target_index+3):

            if takeDay != 1 and not is_numeric_string(text_list[i]):
                break

            if is_numeric_string(text_list[i]) and takeDay != 1:
                takeCount = takeDay
                takeDay = text_list[i]
                break

            if not isTakeCount and is_numeric_string(text_list[i]):
                takeCount = text_list[i]
                isTakeCount = True
                continue

            if isTakeCount and is_numeric_string(text_list[i]):
                takeDay = text_list[i]
                continue


        print("takeCount: {}  takeDay: {}".format(takeCount, takeDay))
        print()
        result.append(
            {
                'pillName': item['pill'],
                'takeCount': takeCount,
                'takeDay': takeDay,
                'takePillTimeList': findTakePillTime(takeCount)
            }
        )


    print(result)

    return result


# Press the green button in the gutter to run the script.
def ocrProcess(image):
    print("text 추출 시작")
    # 언어 설정 (English: en, 한국어: ko 등)
    language = ['en', 'ko']

    # EasyOCR 객체 생성
    reader = easyocr.Reader(language, gpu=False)

    # 이미지에서 텍스트 추출
    results = reader.readtext(image)

    # 점수의 임계값 설정
    threshold = 0.4

    # 결과를 텍스트별로 리스트에 담기
    text_list = [result[1].replace(' ', '').replace('O', '0')
                 for result in results if result[2] > threshold]

    # 결과 출력
    for index, text in enumerate(text_list):
        print('{0}: {1}'.format(index, text))

    return text_list
