import base64

from flask import Flask, request, jsonify

from ocr.service.OCRService import processOCRService

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def receive_data():
        # JSON 데이터 파싱
        json_data = request.get_json()

        # 이미지 데이터 추출
        encoded_image = json_data.get('image', '')

        image_data = base64.b64decode(encoded_image)

        result_data = processOCRService(image_data)

        print(result_data)

        response = jsonify(result_data)

        response.headers['Content-Type'] = 'application/json; charset=utf-8'

        return response

@app.route('/test', methods=['POST'])
def upload_file():
    file = request.files['file']  # 'file'은 Spring에서 전송하는 파일 필드 이름에 맞춰야 합니다.
    # 파일 처리 로직 작성
    file.save('../ocr_image/' + file.filename)  # 파일 저장 예시
    print(file)
    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)