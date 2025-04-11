import eventlet
eventlet.monkey_patch()  # 모든 모듈 임포트 전에 호출

import requests
import time
import logging
from flask import Flask, jsonify, render_template, Response
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import io

###########################
####### mp3파일 STT ########
###########################
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





def get_response_from_model_stream(message):
    url = "http://localhost:5006/order"  # 모델 서버의 로컬 주소

    payload = {
        "header": {
            "interfaceID": "AI-SDC-CAT-001"
        },
        "body": {
            "text": message  # 인식된 텍스트를 모델 서버로 전송
        }
    }

    headers = {
        'Content-Type': 'application/json; charset=utf-8'
    }

    # 모델 서버로 요청을 보내고, 스트리밍 응답을 받음
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        if response.status_code == 200:
            for chunk in response.iter_lines():
                if chunk:
                    line = chunk.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[len('data: '):]
                    yield line
        else:
            print(f"Error: {response.status_code}")


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')  # WebSocket 설정

# WebSocket 이벤트 처리
@socketio.on('message')
def handle_message(data):
    data = request.get_json()
    html_message = data.get('message', '')
    print('html_message:', html_message)
    
    user_message = recognize_speech_from_audio(input_file)
    print("user_message:", user_message)

    # 모델 서버로부터 스트리밍 응답을 받아와 클라이언트로 전송
    for token in get_response_from_model_stream(user_message):
        emit('message', {'token': token})

    # 마지막으로 스트림이 끝났음을 알리는 종료 메시지 전송
    emit('message', {'token': 'END'})

@app.route('/')
def index():
    return render_template("index_mp3_streaming_tts.html")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)

# flask --app response run --host=0.0.0.0 --port=5001