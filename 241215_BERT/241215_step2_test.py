import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pandasql import sqldf

# === 1. 학습된 모델 및 토크나이저 로드 === #
class Args:
    def __init__(self):
        self.output_dir = "/home/user09/beaver/data/shared_files/241215_BERT/checkpoint/klue_roberta_large"
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

# === 2. 데이터프레임 생성 === #
menu_data = pd.DataFrame({
    "name": ["아메리카노", "라떼", "카푸치노"],
    "quantity": [10, 5, 3],
    "price": [3000, 4000, 4500]
})


# === 3. SQL 쿼리를 위한 헬퍼 함수 === #
def run_sql(query, data_frame):
    """pandasql을 사용하여 SQL 실행"""
    from pandasql import sqldf
    pysqldf = lambda q: sqldf(q, {"menu_data": data_frame})
    return pysqldf(query)


# === 4. 의도 분류 함수 === #
def classify_intent(text):
    """학습된 모델을 사용해 의도를 분류"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    predicted_class = logits.argmax(axis=-1).item()
    return predicted_class  # 의도 번호 반환

# === 5. 응답 생성 함수 === #
def generate_response(intent_id, text, menu_data):
    """
    의도에 따라 적절한 응답 생성
    """
    # 템플릿 가져오기
    templates = args.response_templates[intent_id]

    # 다중 템플릿 처리: 랜덤 선택
    if isinstance(templates, list):
        template = random.choice(templates)
    else:
        template = templates

    # 응답 데이터 동적 생성 (예제 데이터 사용)
    response_data = {
        "store_name": "ChatGPT 카페",
        "category1": "커피",
        "category2": "디저트",
        "category3": "음료",
        "popular_item": "아메리카노",
        "new_item": "바닐라 라떼",
        "recommended_item": "카라멜 마끼아또",
        "badge1": "계절 한정",
        "item1": "자몽 에이드",
        "item2": "딸기 라떼",
        "item3": "망고 스무디",
        "regions":'신사동',
        "dine_in": "매장 식사",
        "takeout": "포장",
        "delivery": "배달",
        "payment1": "현금",
        "payment2": "신용카드",
        "payment3": "모바일 페이",
        "open_hour": 9,
        "open_minute": 0,
        "close_hour": 22,
        "close_minute": 0,
        "holiday": "일요일",
        "holiday_status": "영업합니다",
        "region1": "서울",
        "region2": "부산",
        "region3": "대전",
        "region4": "대구",
        "delivery_fee_min": 1000,
        "delivery_fee_max": 3000,
        "minimum_order": 15000,
        "store_address": "서울특별시 강남구 테헤란로 123",
        "total_seats": 50,
        "total_tables": 20,
        "private_room": "프라이빗 룸",
        "outdoor_seats": "야외 테라스",
        "room_count": 3,
        "points_required": 1000,
        "points_percentage": 10,
        "stamps_required": 10,
        "coupon_reward": "아메리카노 무료 쿠폰",
        "event1": "1+1 행사",
        "event2": "여름 할인",
        "event3": "멤버십 가입 혜택",
        "period1": "8월 말",
        "period2": "9월 중순",
        "period3": "12월 말"
    }

    # SQL 데이터 삽입 예시
    if intent_id in [0, 1, 8]:  # SQL이 필요한 의도에 한해 처리
        query = "SELECT name, quantity FROM menu_data WHERE quantity > 0"
        result = run_sql(query, menu_data)  # menu_data를 직접 전달
        print(result)
        response_data.update({"menu_sql_result": result.to_dict(orient="records")})
        print(response_data)

    # 템플릿 채우기
    try:
        response = template.format(**response_data)
    except KeyError as e:
        response = f"필요한 데이터가 누락되었습니다: {e}"

    return response


# === 6. 전체 실행 예제 === #
if __name__ == "__main__":
    user_inputs = [
        # "아메리카노 메뉴 카테고리 알려주세요.",
        # "요즘 인기 메뉴는 뭐예요?",
        "배달 가능한 지역은 어디인가요?"
    ]

    for user_input in user_inputs:
        intent_id = classify_intent(user_input)  # 의도 번호로 분류
        response = generate_response(intent_id, user_input, menu_data)
        print(f"입력: {user_input}")
        print(f"의도: {intent_id}_{args.intents_dict[intent_id]}")
        print(f"응답: {response}")
        print("-" * 50)
