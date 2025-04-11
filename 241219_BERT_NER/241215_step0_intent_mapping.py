import pandas as pd
import re

# 엑셀 파일 읽기
file_path = "/home/user09/beaver/beaver_shared/data/shared_files/241219_BERT_NER/data/intent_v20.xlsx"  # 엑셀 파일 경로를 지정해주세요.
df = pd.read_excel(file_path)

# intents_dict 매핑
intents_dict = {
            0: "메뉴 카테고리 안내",
            1: "특정 상품 및 가격 안내", # 슬롯
            2: "상품에 대한 상세 및 추가 안내",
            3: "(뱃지) 인기메뉴",
            4: "(뱃지) 추천메뉴",
            5: "(뱃지) 대표메뉴",
            6: "(뱃지) 할인, 이벤트 메뉴",
            7: "(뱃지) 1+1 메뉴 문의",
            8: "(뱃지) 신상 메뉴",
            9: "(뱃지) 한정 메뉴",
            10: "(뱃지) 매운맛",
            11: "매장 주문 방식 안내 (키오스크, 테이블오더, 스마트주문 등)",
            12: "주문한 상품 전달 방식 안내 (내점, 포장, 배달 등)",
            13: "결제 방법 안내(현금, 카드, 간편 결제 등)",
            14: "특정 결제 방법 상세 안내", # 슬롯
            15: "결제 방법 추가 안내",
            16: "영업 시간 안내",
            17: "브레이크 타임 안내",
            18: "휴무일 안내 (정기 - 임시휴무/주말/공휴일)",
            19: "휴무일 상세 안내 (정기 - 임시휴무/주말/공휴일)",
            20: "영업시간 및 휴무일 추가 안내",
            21: "특정 요일에 대한 영업 여부", # 슬롯
            22: "배달 가능 지역 안내",
            23: "배달비 및 최소 주문 금액 안내",
            24: "배달 추가 안내",
            25: "테이블 점유 안내",
            26: "테이블 점유 추가 안내",
            27: "상점 주소 안내 및 지도 링크 전달",
            28: "대중교통 이용 방법 안내",
            29: "근처 랜드마크 안내",
            30: "테이블 및 좌석 수 안내",
            31: "단체석 및 예약석 유무 안내",
            32: "야외 테라스 또는 개별 룸 여부 안내",
            33: "상점 규모 및 시설 추가 안내",
            34: "멤버십 가입 안내",
            35: "멤버십 혜택 안내",
            36: "포인트 적립 안내",
            37: "포인트 사용 안내",
            38: "쿠폰 발행 안내",
            39: "쿠폰 사용 안내",
            40: "멤버십, 포인트 및 쿠폰에 대한 추가 안내",
            41: "현재 진행 중인 이벤트 안내",
            42: "현재 진행 중인 이벤트 상세 안내",
            43: "현재 진행 중인 이벤트 추가 안내",
            44: "CallBackIntent",
            45: "감사 인텐트",
            46: "인사 인텐트",
            47: "FallBackIntent"
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
df["user"] = df["user"].apply(lambda x: re.sub(r'["\']', '', str(x).strip()))

# 결과 확인
#print(df)

# 수정된 엑셀 파일 저장
output_path = "/home/user09/beaver/beaver_shared/data/shared_files/241219_BERT_NER/data/user_intent_v14.csv"
df.to_csv(output_path, index=False)
print(f"파일이 성공적으로 저장되었습니다: {output_path}")



### 생성한 Intent 데이터의 Intent 별 개수 세기 ### 
# intent 컬럼의 종류별 개수 계산
intent_counts = df['intent_num'].value_counts()

# 결과 출력
print("Intent 별 개수:")
print(intent_counts)

# 결과를 엑셀 파일로 저장 (선택사항)
output_path = "intent_counts.xlsx"
intent_counts.to_excel(output_path, sheet_name="Intent Counts")
print(f"결과가 성공적으로 저장되었습니다: {output_path}")