import pandas as pd
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pandasql import sqldf

# === 1. 학습된 모델 및 토크나이저 로드 === #
class Args:
    def __init__(self):
        self.output_dir = "C:/Users/user/PycharmProjects/pythonProject/beaver/241215_BERT_XXXXXX/checkpoint/klue_roberta_large"
        self.intents_dict = {
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
        self.response_templates = {
                0: [
                    "저희 {STORE_NM}에서는 {STD_CATEGORY_NM}의 메뉴 종류들이 있어요. 자세한 사항은 보내드린 메뉴판을 참고해주세요."
                ],
                1: [
                    "저희 {STORE_NM}에서 가장 인기 있는 상품은 {TOT_PROD_QTY} 이에요."
                ],
                2: [
                    "{BADGE}에 해당하는 메뉴는 {SPECIAL_PROD}이 있어요."
                ],
                3: [
                    "저희 {STORE_NM}은 {ORDER_TYPE}이 가능 해요."  # ex. 저희 홍콩반점은 매장식사, 포장 배달이 가능해요
                ],
                4: [
                    "저희 {STORE_NM}에서는 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD}으로 결제 가능해요."
                ],
                5: [
                    "저희 {STORE_NM}의 영업시간은 {SALE_BGN_TM}부터 {SALE_END_TM}까지에요."
                ],
                6: [
                    "저희 {STORE_NM}의 휴일은 매주 {HOLIDAY_TYPE_CD}이에요."
                ],
                7: [
                    "저희 {STORE_NM}의 배달지역은 {DLVR_ZONE_RADS} 지역에 배달이 가능 합니다."
                ],
                8: [
                    "저희 {STORE_NM}의 배달 팁은 최소 {DLVR_TIP_AMT_MIN} 부터 최대 {DLVR_TIP_AMT_MAX} 이에요."
                ],
                9: [
                    "저희는 포장 서비스를 {PACK_YN} 있습니다."  # ex. 제공하고/제공하지않
                ],
                10: [
                    "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요."
                ],
                11: [
                    "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요."
                ],
                12: [
                    "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요."
                ],
                13: [
                    "저희 {STORE_NM}의 주소는 {ROAD_NM_ADDR} 에요."
                ],
                14: [
                    "저희 {STORE_NM}의 테이블 수는 {TABLE_NM}개 에요."
                ],
                15: [
                    "저희 {STORE_NM}에는 {ROOM_SEAT}, {OUTDOOR_SEAT}이 각각 {ROOM_TABLE_ZONE_NM}, {OUTDOOR_TABLE_ZONE_NM}개 있어요."
                ],
                16: [
                    "포인트 서비스나 쿠폰을 발급 받기 위해서는 회원 가입이 필요합니다. 회원가입은 스마트 주문 앱 또는 키오스크에서 가입 가능해요."
                ],
                17: [
                    "저희 {STORE_NM} 에서는 포인트 {USE_POSBL_BASE_POINT}점 부터 사용 가능해요."
                ],
                18: [
                    "저희 {STORE_NM} 에서는 스탬프 {STAMP_TMS}개 적립 시 {COUPON_PACK_NM} 쿠폰을 발급해 주고 있어요."
                ],
                19: [
                    "현재 저희 {STORE_NM} 에서는 {EVENT_NM} 이벤트가 진행되고 있으며 {EVENT_BNEF_CND_CD} 동안 진행돼요."
                ],
                20: [
                    "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요."
                ],
                21: [
                    "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요."
                ]
            }



args = Args()

tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터베이스 로딩 함수
def load_excel_to_dataframe(file_path):
    dataframes = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in dataframes.items():
        df.columns = [col.strip().replace(" ", "_") for col in df.columns]
        df.fillna("없음", inplace=True)
    return df

# SQL 실행 함수
def execute_sql(intent_id, df):
    try:
        if intent_id == 0:  # 메뉴 카테고리 안내
            result = df.query("STD_CATEGORY_NM != '없음' and STD_CATEGORY_NM != ''")[['STORE_NM', 'STD_CATEGORY_NM']].drop_duplicates()
        elif intent_id == 1:  # 인기 / 추천 메뉴 안내
            result = df.query("QTY != '없음'")[['STD_PROD_NM', 'QTY']].groupby('STD_PROD_NM').max().sort_values(by='QTY', ascending=False).reset_index()
        elif intent_id == 2:  # 배지 및 특선 메뉴 안내
            result = df.query("BADGE != '없음' and SPECIAL_PROD != '없음'")[['BADGE', 'SPECIAL_PROD']].drop_duplicates()
        elif intent_id == 3:  # 주문 / 전달 방식 안내
            result = df.query("ORDER_TYPE != '없음' and ORDER_TYPE != ''")[['ORDER_TYPE']].drop_duplicates()
        elif intent_id == 4:  # 결제 방법 안내
            result = df.query("PAYMNT_MN_CD != '없음' and EASY_PAYMNT_TYPE_CD != '없음'")[['PAYMNT_MN_CD', 'EASY_PAYMNT_TYPE_CD']].drop_duplicates()
        elif intent_id == 5:  # 영업 시작/종료 시간 안내
            result = df.query("SALE_BGN_TM != '없음' and SALE_END_TM != '없음'")[['SALE_BGN_TM', 'SALE_END_TM']].drop_duplicates()
        elif intent_id == 6:  # 휴일 정보 안내
            print(df)
            result = df.query("HOLIDAY_TYPE_CD != '없음' and HOLIDAY_TYPE_CD != ''")[['HOLIDAY_TYPE_CD']].drop_duplicates()
        elif intent_id == 7:  # 배달 가능 지역 안내
            result = df.query("DLVR_ZONE_RADS != '없음' and DLVR_ZONE_RADS != ''")[['DLVR_ZONE_RADS']].drop_duplicates()
        elif intent_id == 8:  # 배달비 및 최소 주문 금액 안내
            result = df.query("DLVR_TIP_AMT_MIN != '없음' and DLVR_TIP_AMT_MAX != '없음'")[['DLVR_TIP_AMT_MIN', 'DLVR_TIP_AMT_MAX']].drop_duplicates()
        elif intent_id == 9:  # 포장 서비스 제공 여부 안내
            result = df.query("PACK_YN != '없음'")[['PACK_YN']].drop_duplicates()
        elif intent_id == 13:  # 상점 주소 안내
            result = df.query("ROAD_NM_ADDR != '없음' and ROAD_NM_ADDR != ''")[['STORE_NM', 'ROAD_NM_ADDR']].drop_duplicates()
        elif intent_id == 14:  # 테이블 및 좌석 안내
            result = df.query("TABLE_NM != '없음'")[['TABLE_NM', 'ROOM_SEAT', 'OUTDOOR_SEAT']].drop_duplicates()
        elif intent_id == 15:  # 룸 및 야외 테이블 안내
            result = df.query("ROOM_TABLE_ZONE_NM != '없음' and OUTDOOR_TABLE_ZONE_NM != '없음'")[['ROOM_TABLE_ZONE_NM', 'OUTDOOR_TABLE_ZONE_NM']].drop_duplicates()
        elif intent_id == 17:  # 포인트 사용 기준 안내
            result = df.query("USE_POSBL_BASE_POINT != '없음'")[['USE_POSBL_BASE_POINT']].drop_duplicates()
        elif intent_id == 18:  # 스탬프 및 쿠폰 안내
            result = df.query("STAMP_TMS != '없음' and COUPON_PACK_NM != '없음'")[['STAMP_TMS', 'COUPON_PACK_NM']].drop_duplicates()
        elif intent_id == 19:  # 이벤트 및 할인 혜택 안내
            result = df.query("EVENT_NM != '없음' and EVENT_BNEF_CND_CD != '없음'")[['EVENT_NM', 'EVENT_BNEF_CND_CD']].drop_duplicates()
        elif intent_id in [10, 11, 12, 16, 20, 21]:  # 고정 응답이 필요한 intent_id
            print(f"Static response required for intent_id {intent_id}")
            return None
        else:
            print("해당 intent에 대한 처리가 없습니다.")
            return None

        return result.iloc[0].to_dict() if not result.empty else None

    except Exception as e:
        print(f"쿼리 실행 오류: {e}")
        return None

if __name__ == "__main__":
    # user_input = input("사용자 발화를 입력하세요: ")
    user_input = "휴일"
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    intent_id = outputs.logits.argmax().item()
    print("인텐트:", intent_id)
    dataframes = load_excel_to_dataframe("C:/Users/user/PycharmProjects/pythonProject/beaver/241215_BERT/data/dataset_SQL_general_ju_1_105200_preprocessed.xlsx")
    
    data = execute_sql(intent_id, dataframes)#[list(dataframes.keys())[0]])
    print(data)

    templates = args.response_templates[intent_id]

    # 다중 템플릿 처리: 랜덤 선택
    if isinstance(templates, list):
        template = random.choice(templates)
    else:
        template = templates
        
    if data:
        response = template.format(**data)
    else:
        response = template
    print(f"응답: {response}")