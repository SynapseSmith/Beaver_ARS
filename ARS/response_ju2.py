import eventlet
eventlet.monkey_patch()

import requests
from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

def get_response_from_model_stream(message):
    url = "http://localhost:5006/order"

    payload = {
        "header": {
            "interfaceID": "AI-SDC-CAT-001"
        },
        "body": {
            "text": message
        }
    }

    headers = {
        'Content-Type': 'application/json; charset=utf-8'
    }

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

@socketio.on('message')
def handle_message(data):
    print("클라이언트로부터 받은 데이터:", data)
    user_message = data.get('message', '')
    print("추출된 user_message:", user_message)
    # 백그라운드 작업 시작
    socketio.start_background_task(target=process_message, user_message=user_message, sid=request.sid)

def process_message(user_message, sid):
    # 모델 서버로부터 스트리밍 응답을 받아와 클라이언트로 전송
    for token in get_response_from_model_stream(user_message):
        socketio.emit('message', {'token': token}, room=sid)
    # 스트림이 끝났음을 알리는 종료 메시지 전송
    socketio.emit('message', {'token': 'END'}, room=sid)

@app.route('/')
def index():
    return render_template("index_ju2.html")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5011)
