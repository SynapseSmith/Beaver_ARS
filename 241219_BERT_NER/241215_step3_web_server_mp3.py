import speech_recognition as sr
import requests
import time
import logging
from RealtimeTTS import TextToAudioStream, GTTSEngine
from flask import Flask, jsonify, request, render_template

def get_response_from_local_model(recognized_text):
   
    url = "http://localhost:5050/order"  # 클라이언트가 요청을 보내는 주소(로컬 서버 주소)
    
    # 서버에 보낼 페이로드
    payload = {
        "header": {
            "interfaceID": "AI-SDC-CAT-001"
        },
        "body": {
            "text": recognized_text  # 인식된 텍스트를 서버로 전송
        }
    }
    
    headers = {
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    
    print("모델의 인풋:", recognized_text)
    response = requests.post(url, json=payload, headers=headers)
    
    # 서버 응답 상태 코드와 내용 출력
    print("서버 응답 상태 코드:", response.status_code)  
    if response.status_code == 200:
        try:
            server_response = response.json()  # JSON 응답 파싱
            return server_response.get("response", "응답 없음")  # 'response' 키 값 추출
        except Exception as e:
            print(f"JSON 파싱 오류: {e}")
            return "JSON 파싱 오류가 발생했습니다."
    else:
        return f"서버 오류: {response.status_code}"


app = Flask(__name__)
@app.route('/get-response', methods=['POST'])
def generate():    
    data = request.get_json()
    user_message = data.get('message', '')
    print("user_message:", user_message)
    
    query = get_response_from_local_model(user_message)
    print('query:', query)
    
    return jsonify({'response': query})
 
# HTML 페이지 렌더링
@app.route('/')
def index():
    return render_template("main_index_polly_copy.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=True)