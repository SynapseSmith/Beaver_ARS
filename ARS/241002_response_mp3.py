import speech_recognition as sr
import requests
import time
import logging
from RealtimeTTS import TextToAudioStream, GTTSEngine
from flask import Flask, jsonify, request, render_template
from pydub import AudioSegment
import io

input_file = "/home/user09/beaver/data/shared_files/ARS/static/audio/주차정보알려줘2.mp3"

def convert_mp3_to_wav(mp3_file):  # mp3 -> wav
    audio = AudioSegment.from_mp3(mp3_file)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


def recognize_speech_from_audio(mp3_file):  # STT
    recognizer = sr.Recognizer()
    wav_data = convert_mp3_to_wav(mp3_file)
    with sr.AudioFile(wav_data) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="ko-KR")
        print(f"변환결과: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")



def get_response_from_local_model(recognized_text):
    # url = "http://100.104.18.53:5000/order"
    # url = "https://80f0-203-229-206-42.ngrok-free.app/order"  #클라이언트가 요청을 보내는 주소(로컬 서버 주소)
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
    print("mp3파일 인식 결과:", recognized_text)
    response = requests.post(url, json=payload, headers=headers)
    print("서버 응답 상태 코드:", response.status_code)  # 응답 상태 코드 출력
    print("서버 응답 내용:", response.text)  # 응답 본문 출력
    return response.text


app = Flask(__name__)
@app.route('/get-response', methods=['POST'])
def generate():    
    data = request.get_json()
    html_message = data.get('message', '')
    print('html_message:', html_message)
    
    user_message = recognize_speech_from_audio(input_file)
    print("user_message:", user_message)
    
    query = get_response_from_local_model(user_message)
    print('query:', query)
    return jsonify({'response': query})
 
# HTML 페이지 렌더링
@app.route('/')
def index():
    return render_template("241002_index_mp3.html")   # templates 폴더명을 기본으로 보고, 그 안에 있는 html 파일 인식

if __name__ == '__main__':
    app.run(debug=True)
    
# flask --app 241002_response_mp3 run --host=0.0.0.0 --port=5001  ==> 외부 요청을 받을 떄 사용하는 주소..?
# https://9a2d-203-229-206-42.ngrok-free.app