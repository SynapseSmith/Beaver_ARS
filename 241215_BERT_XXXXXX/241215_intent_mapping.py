import pandas as pd

# 엑셀 파일 읽기
file_path = "C:/Users/user/PycharmProjects/pythonProject/beaver/241215_BERT_XXXXXX/data/intent.xlsx"  # 엑셀 파일 경로를 지정해주세요.
df = pd.read_excel(file_path)

# intents_dict 매핑
intents_dict = {
    0: "메뉴 카테고리 안내",
    1: "인기 / 추천 / 신메뉴 안내",
    2: "계절 한정 메뉴 (예: 여름 특선, 겨울 메뉴) / 프로모션 메뉴 (예: 신메뉴 할인) 안내",
    3: "주문 / 전달 방식 안내 (키오스크, 테이블오더, 스마트주문 / 내점, 포장, 배달 등)",
    4: "결제 방법 안내(현금, 카드, 간편 결제등)",
    5: "영업 시작/종료 시간 안내",
    6: "정기 휴무/주말/공휴일 운영 여부 안내",
    7: "배달 가능 지역 안내",
    8: "배달비 및 최소 주문 금액 안내",
    9: "포장 서비스 제공 여부 안내",
    10: "예약 가능 여부 안내(전화/온라인)",
    11: "예약 취소/변경 절차 안내",
    12: "고객 대기 및 혼잡도 안내",
    13: "상점 주소 안내 및 지도 링크 전달",
    14: "테이블 배치 및 좌석 수 안내",
    15: "야외 테라스 또는 개별 룸 여부 안내",
    16: "멤버십 가입 / 혜택 안내",
    17: "포인트 적립 / 사용 관련 안내",
    18: "쿠폰 발행 / 사용 안내",
    19: "현재 진행 중인 이벤트 및 할인 혜택, 기간 정보 안내",
    20: "상점 관리자 연결 안내",
    21: "fallbackintent"
}

# intent 값을 숫자로 매핑하여 intent_num 생성
intent_num_mapping = {v: int(k) for k, v in intents_dict.items()}

# 매핑 작업 수행
df["intent_num"] = df["intent"].map(intent_num_mapping)

# NaN 값이 있는지 확인
if df["intent_num"].isnull().any():
    print("다음 값들은 매핑되지 않았습니다:")
    print(df[df["intent_num"].isnull()])
else:
    print("모든 intent 값이 성공적으로 매핑되었습니다.")

# NaN 제거 및 데이터 타입 변경
# df["intent_num"] = df["intent_num"].fillna(-1).astype(int)  # NaN 값을 -1로 대체 후 정수형 변환

# intent 컬럼 삭제
df = df.drop(columns=["intent"])

# user 컬럼에서 양쪽 따옴표 제거
df["user"] = df["user"].str.strip('""')

# 결과 확인
#print(df)

# 수정된 엑셀 파일 저장
output_path = "C:/Users/user/PycharmProjects/pythonProject/beaver/241215_BERT_XXXXXX/data/user_intent.xlsx"
df.to_excel(output_path, index=False)
print(f"파일이 성공적으로 저장되었습니다: {output_path}")
