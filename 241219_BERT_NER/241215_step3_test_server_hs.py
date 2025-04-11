import requests
import json

datas = {
    'header' : {
        'interfaceID': 'AI-SDC-CAT-001',
        'interfaceMsg' : '메뉴 주문 추론',
        'storeNo': '123456'
    },
    #'inputTranscript': '죄송한데, 영업 시간을 알 수 있을까요?'   
    "body": {
        "text": "고추짜장은 얼마야?" # 인식된 텍스트를 서버로 전송
        }
}

headers = {
    "Content-Type": "application/json"
}

# 우회주소 비버웍스에서 사용중이므로 테스트에 사용하지 말 것!
# url = 'https://c8db-203-229-206-42.ngrok-free.app/order'
# url = 'https://dd62495328fa.ngrok.app/order'
url = 'http://127.0.0.1:5060/order'

# JSON 직렬화 후 요청 보내기
datas = json.dumps(datas)
response = requests.post(url, data=datas, headers=headers)

# JSON 응답에서 response 값 추출 및 출력
server_return = response.json()  # 응답 JSON 파싱
print(server_return.get("response"))  # 'response' 키의 값만 출력