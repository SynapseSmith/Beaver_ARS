##########################################
###########원본 파일 수정 주의하기##########
##########################################
import speech_recognition as sr
import requests
import time
import logging
from RealtimeTTS import TextToAudioStream, GTTSEngine
from flask import Flask, jsonify, request, render_template

def get_response_from_local_model(recognized_text):
    # url = "http://100.104.18.53:5000/order"
    # url = " https://9a2d-203-229-206-42.ngrok-free.app"  #클라이언트가 요청을 보내는 주소(로컬 서버 주소)
    ########
    # [민지]: 웹 서버랑 모델 서버랑 같은 서버 내에 있어서 80f0 주소를 빼고 로컬호스트로 할당했슴다!! 바꾸지 말아주세용 흑흑
    url = "http://localhost:5006/order"  #클라이언트가 요청을 보내는 주소(로컬 서버 주소)
    ########
    
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
    print("서버 응답 상태 코드:", response.status_code)  # 응답 상태 코드 출력
    print("서버 응답 내용:", response.text)  # 응답 본문 출력
    return response.text


app = Flask(__name__)
@app.route('/get-response', methods=['POST'])
def generate():    
    data = request.get_json()
    user_message = data.get('message', '')
    print("user_message:", user_message)
    query = get_response_from_local_model(user_message)
    print('query:', query)
    
    # # 각 토큰을 스트리밍으로 클라이언트에 전달
    # for chunk in query:
    #     text = chunk['choices'][0]['delta'].get('content', '')
    #     yield f"data: {text}\n\n"  # 서버-전송 이벤트 (SSE) 형식
    
    # for chunk in query:
    #     # chunk에서 delta와 content를 추출
    #     if 'choices' in chunk and 'delta' in chunk['choices'][0]:
    #         content = chunk['choices'][0]['delta'].get('content')
    #         if content:  # content가 존재할 때만 전송
    #             yield f"data: {content}\n\n"
    #     else:
    #         yield f"data: {chunk}\n\n"
    return jsonify({'response': query})
 
# HTML 페이지 렌더링
@app.route('/')
def index():
    return render_template("index_realtimetts.html")
if __name__ == '__main__':
    app.run(debug=True)
    
# flask --app response run --host=0.0.0.0 --port=5001  ==> 외부 요청을 받을 떄 사용하는 주소..?
# https://9a2d-203-229-206-42.ngrok-free.app