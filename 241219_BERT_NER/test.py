import requests
import json
from tqdm import tqdm
import pandas as pd

url = 'https://f523bcd40861.ngrok.app/order'

test_txt = pd.read_csv("/home/user07/beaver/beaver_shared/ARS_TEST_DATA.csv")
test_txt = test_txt.head(2)
txt_response = []
txt_response_times = []  # 응답 시간을 저장할 리스트

# mp3 파일 경로 설정 및 파일 전송을 for문으로 처리
for i in tqdm(range(len(test_txt)), desc="Processing TXT files"):  # 진행 상황 표시

    payload = {
        'header' : {
            'interfaceID': 'IF-ARS-CHAT-001',
            'interfaceMsg' : '질문 추론'
        },
        'body': {
            'text' : test_txt['question'][i]
        }        
    }


    # 텍스트 보낼 때 주석 해제
    payload = json.dumps(payload)
    reponse = requests.post(url, data=payload)

    server_return = reponse.json()
    txt_response.append(server_return.get("response"))
 
    # 응답 시간 저장
    txt_response_times.append(server_return.get("response"))

# mp3_response와 response_times를 test_txt DataFrame에 추가
test_txt['chatbot_response'] = txt_response
test_txt['txt_time'] = txt_response_times
print(test_txt)


print('txt test : done')