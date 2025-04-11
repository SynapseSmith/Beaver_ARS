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
        "text": "차량은 몇대까지 주차 가능한가요" # 인식된 텍스트를 서버로 전송
        }
}

headers = {
}

# 우회주소 비버웍스에서 사용중이므로 테스트에 사용하지 말 것!
# url = 'https://c8db-203-229-206-42.ngrok-free.app/order'
url = 'http://127.0.0.1:6000/order'

datas = json.dumps(datas)
reponse = requests.post(url, data=datas, headers=headers)
# server_return = reponse.json()
server_return = reponse.text
print('-'*50)
print(server_return)
print('-'*50)