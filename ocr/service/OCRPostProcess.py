import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sympy.physics.quantum.identitysearch import np


def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [row[0] for row in reader]


def get_similar_range_pill_names(query, pill_names, index_list, n):
    # 알약 이름 데이터셋 벡터화
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n, n+1),  lowercase=False)

    # 알약 이름 데이터셋 벡터화
    pill_names_tfidf = vectorizer.fit_transform(pill_names['name'])

    # 검색어 벡터화
    query_tfidf = vectorizer.transform([query])

    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(query_tfidf, pill_names_tfidf).flatten()

    # 유사도가 가장 높은 인덱스 구하기
    similar_indices = cosine_similarities.argsort()[:-5:-1]

    # similar_indices = cosine_similarities.argsort([])

    similar_pill_names = []

    # 유사한 알약 이름들 출력
    for i in similar_indices:
        similar_pill_names.append(pill_names.iloc[i]['name'])

    max_index = similar_indices[0]
    return {'name': similar_pill_names, 'similarities': [np.max(cosine_similarities)], 'index': [index_list[similar_indices[0]]]}


def get_similar_pill_names(query, pill_names, n):

    # 알약 이름 데이터셋 벡터화
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n, n+1),  lowercase=False)

    # 알약 이름 데이터셋 벡터화
    pill_names_tfidf = vectorizer.fit_transform(pill_names['name'])

    # 검색어 벡터화
    query_tfidf = vectorizer.transform([query])

    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(query_tfidf, pill_names_tfidf).flatten()

    # 유사도가 가장 높은 인덱스 구하기
    similar_indices = cosine_similarities.argsort()[:-5:-1]

    # similar_indices = cosine_similarities.argsort([])

    similar_pill_names = []

    # 유사한 알약 이름들 출력
    for i in similar_indices:
        similar_pill_names.append(pill_names.iloc[i]['name'])

    return {'name': similar_pill_names, 'similarities': [np.max(cosine_similarities)], 'index': similar_indices}

# 이름 유사도 분석
def find_similar_person_names(data_set, query):
    # 유사한 알약 이름 출력
    similar_person_name = get_similar_pill_names(query, data_set, 1)
    similar_person_name_similarities = np.max(similar_person_name['similarities'])
    if similar_person_name_similarities > 0.65:
        return ''

    return similar_person_name


# 글자 범위를 다르게한 유사도 검사 연속 수행 함수
def find_similar_pill_names(data_set, query):

    # 유사한 알약 이름 출력
    similar_pill_name = get_similar_pill_names(query, data_set, 1)
    similar_pill_name_similarities = np.max(similar_pill_name['similarities'])
    similar_pill_name_index_list = similar_pill_name['index']

    if similar_pill_name_similarities >= 0.99:
        return similar_pill_name

    new_data_set = similar_pill_name['name']
    new_data_set = pd.DataFrame({'name': new_data_set})

    new_similar_pill_name = get_similar_range_pill_names(query, new_data_set, similar_pill_name_index_list, 2)
    new_similar_pill_name_similarities = np.max(new_similar_pill_name['similarities'])


    if similar_pill_name_similarities > new_similar_pill_name_similarities:
        return similar_pill_name

    return new_similar_pill_name

# 앞에 등장하는 () 삭제
def remove_bracket(data):
    regex = re.compile(r'\((.*?)\)')  # 정규표현식에서 괄호 안의 문자를 찾음
    match = regex.search(data)

    if match:
        data = data[:match.start()] + data[match.end():]  # 괄호 부분을 삭제하여 문자열을 재조합함

    return data


# 한글 밀리그램을 영어 mg 로 변환
def chang_korea_mg_to_english(data):
    pattern = r'(\d+(\.\d+)?)'
    regex = re.compile(pattern)
    matches = regex.findall(data)

    for match in matches:
        mg = match[0]
        mg_eng = f'{mg}mg'
        mg_kor = f'{mg}밀리그람'
        data = data.replace(mg_kor, mg_eng)

        mg_kor = f'{mg}밀리그램'
        data = data.replace(mg_kor, mg_eng)

        mg_kor = f'{mg}미리그람'
        data = data.replace(mg_kor, mg_eng)

        mg_kor = f'{mg}미리그램'
        data = data.replace(mg_kor, mg_eng)
    return data

# mg 를 한글로 변환
def chang_english_mg_to_korea(data, korea):
    pattern = r'(\d+(\.\d+)?)'
    regex = re.compile(pattern)
    matches = regex.findall(data)

    for match in matches:
        mg = match[0]
        mg_eng = f'{mg}mg'
        mg_kor = f'{mg}{korea}'
        data = data.replace(mg_eng, mg_kor)
    return data

# 한글 개수 count
def count_korean_chars(text):
    count = 0
    for char in text:
        if '가' <= char <= '힣':
            count += 1
    return count

# mg 를 제외한 영어 제거
def remove_english_except_mg(data):
    remove_english = remove_english_words(data)
    if len(remove_english) < 2:
        return ''
    return replace_m_with_mg(remove_english_except_m(remove_english))

def remove_english_except_m(text):
    pattern = re.compile(r'(?<![0-9])[a-zA-Z]{2}')
    return re.sub(pattern, '', text)

def replace_m_with_mg(text):
    pattern = re.compile(r'm\w')
    result = re.sub(pattern, 'm', text)

    pattern = re.compile(r'm(?=\w)')
    result = re.sub(pattern, 'mg', result)
    return result

def remove_english_words(text):
    pattern = r'\b[A-Za-z]+\b'  # 영어로만 이루어진 단어를 찾기 위한 정규표현식 패턴
    result = re.sub(pattern, '', text)  # 정규표현식에 맞는 단어를 삭제
    return result

# 숫자 제거
def remove_number(data):
    return re.sub(r'\d+', '', data)

# 단어에서 mg 제거
def remove_mg_number(data):
    # 숫자 제거
    result =remove_number(data)
    # "mg" 제거
    result = result.replace("mg", "")
    return result


# 여러 파실 수행
def parsing(data):

    if count_korean_chars(data) < 3:
        return ''

    # 숫자로 이뤄진 텍스트 제거
    if len(remove_number(data)) == 0:
        return ''

    # 영어로 이뤄진 텍스트 제거
    if len(remove_english_except_mg(data)) == 0 or len(data) <= 2:
        return ''

    # 번호 보험 병원 숫자 기호 중 한 개라도 들어가면 제거
    keywords = ["번호", "보험", "환자", "약국", "병원", "숫자", "tab", "cap", "기호", "용법", "유법", "총량", "명칭", "기타", "조제약사"]

    for keyword in keywords:
        if keyword in data:
            return ''

    if(len(data) < 6):
        name_data_set = read_csv("../data_set/name_data.csv")
        person_names = pd.DataFrame({'name': name_data_set})
        english_similar_pill = find_similar_person_names(person_names, data)
        if english_similar_pill == '':
            return ''

    parsing_data = remove_bracket(data.replace(" ", ""))
    parsing_data = chang_korea_mg_to_english(parsing_data)

    return parsing_data

# OCR 리스트 데이터를 파싱해서 알약 이름으로 변경하는 함수
def start_similar_pill_names(data_set, ocr_data_list):
    pill_data = read_csv("../data_set/pillName.csv")
    pill_data_set = pd.DataFrame({'name': pill_data})
    # 알약 이름 데이터셋 구성
    pill_names = pd.DataFrame({'name': data_set})

    result_list = []
    for data in ocr_data_list:

        parsing_data = parsing(data)

        if len(parsing_data) == 0:
            result_list.append('')
            continue

        # 기본 data의 유사도
        english_similar_pill = find_similar_pill_names(pill_names, parsing_data)

        if english_similar_pill == '':
            result_list.append('')
            continue

        max_similaritie = english_similar_pill['similarities'][0]
        find_pill_name = english_similar_pill['name'][0]
        pill_num = english_similar_pill['index'][0]

        # 밀리그람 또는 mg가 포함된 약인 경우
        if 'mg' in parsing_data:
            # mg가 없는 경우 유사도

            # mg 와 숫자 모두 제거
            none_mg_parsing_data = remove_mg_number(parsing_data)

            # pill_names에서 none_mg_parsing_data와 길이가 같거나 큰 문자열만 필터링
            filtered_pill_names = pill_names[pill_names['name'].apply(lambda x: len(x) >= len(none_mg_parsing_data))]

            none_mg_similar_pill = find_similar_pill_names(filtered_pill_names, none_mg_parsing_data)
            none_mg_similaritie = none_mg_similar_pill['similarities'][0]
            if none_mg_similaritie > max_similaritie:
                max_similaritie = none_mg_similaritie
                find_pill_name = none_mg_similar_pill['name'][0]
                pill_num = none_mg_similar_pill['index'][0]

            # 밀리그람인 경우 유사도
            korea1_parsing_data = chang_english_mg_to_korea(parsing_data, '밀리그람')
            korea1_similar_pill = find_similar_pill_names(pill_names, korea1_parsing_data)
            korea1_similaritie = korea1_similar_pill['similarities'][0]
            if korea1_similaritie > max_similaritie:
                max_similaritie = korea1_similaritie
                find_pill_name = korea1_similar_pill['name'][0]
                pill_num = korea1_similar_pill['index'][0]

            # 밀리그램 경우 유사도
            korea2_parsing_data = chang_english_mg_to_korea(parsing_data, '밀리그램')
            korea2_similar_pill = find_similar_pill_names(pill_names, korea2_parsing_data)
            korea2_similaritie = korea2_similar_pill['similarities'][0]
            if korea2_similaritie > max_similaritie:
                max_similaritie = korea2_similaritie
                find_pill_name = korea2_similar_pill['name'][0]
                pill_num = korea2_similar_pill['index'][0]

            # 미리그람 경우 유사도
            korea3_parsing_data = chang_english_mg_to_korea(parsing_data, '미리그람')
            korea3_similar_pill = find_similar_pill_names(pill_names, korea3_parsing_data)
            korea3_similaritie = korea3_similar_pill['similarities'][0]
            if korea3_similaritie > max_similaritie:
                max_similaritie = korea3_similaritie
                find_pill_name = korea3_similar_pill['name'][0]
                pill_num = korea3_similar_pill['index'][0]

            # 미리그램 경우 유사도
            korea4_parsing_data = chang_english_mg_to_korea(parsing_data, '미리그램')
            korea4_similar_pill = find_similar_pill_names(pill_names, korea4_parsing_data)
            korea4_similaritie = korea4_similar_pill['similarities'][0]
            if korea4_similaritie > max_similaritie:
                max_similaritie = korea4_similaritie
                find_pill_name = korea4_similar_pill['name'][0]
                pill_num = korea4_similar_pill['index'][0]


        if max_similaritie < 0.35:
            find_pill_name = ''

        # print("텍스트 '{0}'과 가장 유사한 알약 = {1} 유사도={2}".format(data, find_pill_name, max_similaritie))
        if find_pill_name != '':
            result_list.append(pill_data_set.iloc[pill_num]['name'])
        else:
            result_list.append("")

    return result_list

def ocrPostProcess(text_list):

    # 알약 이름 데이터셋 구성
    data_set = read_csv("../data_set/updatePill.csv")
    data = text_list
    find_data = start_similar_pill_names(data_set, data)

    result = []
    for index, item in enumerate(find_data):
        if item != "":
            result.append({"index": index, 'pill': item})

    return result

