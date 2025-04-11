import pandas as pd
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pandasql import sqldf

# === 1. 학습된 모델 및 토크나이저 로드 === #
class Args:
    def __init__(self):
        self.output_dir = "/home/user09/beaver/data/shared_files/241215_BERT/checkpoint/klue_roberta_large"
        self.intents_dict = {
            0: "메뉴 카테고리 안내", 
            1: "인기 / 추천 / 신메뉴 안내",  # QTY
            2: "계절 한정 메뉴 (예: 여름 특선, 겨울 메뉴) / 프로모션 메뉴 (예: 신메뉴 할인) 안내",   # 
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
                "저희 {STORE_NM}에서는 {STD_CATEGORY_NM}의 메뉴를 제공합니다. 메뉴판을 확인해주세요.",
                "저희 {STORE_NM}에서는 {STD_CATEGORY_NM}의 다양한 메뉴가 준비되어 있습니다.",
                "저희 {STORE_NM}의 {STD_CATEGORY_NM} 메뉴를 소개해 드릴게요. 메뉴판을 참조하세요.",
                "{STORE_NM}에서는 {STD_CATEGORY_NM}와 같은 메뉴를 판매하고 있습니다."
            ],
            1: [
                "저희 {STORE_NM}에서 가장 인기 있는 상품은 {TOT_PROD_QTY} 이에요.",
                "{STORE_NM}에서 고객님들이 가장 많이 찾는 메뉴는 {TOT_PROD_QTY} 입니다.",
                "{TOT_PROD_QTY}는 {STORE_NM}에서 가장 많은 사랑을 받고 있는 메뉴입니다.",
                "저희 {STORE_NM}에서 가장 인기 있는 메뉴는 바로 {TOT_PROD_QTY} 입니다."
            ],
            2: [
                "{BADGE}에 해당하는 메뉴는 {SPECIAL_PROD}이 있어요.",
                "저희 {STORE_NM}의 {BADGE} 메뉴는 {SPECIAL_PROD}입니다.",
                "{BADGE}로 추천드리는 메뉴는 {SPECIAL_PROD}입니다.",
                "{BADGE} 메뉴 중 하나는 {SPECIAL_PROD}입니다."
            ],
            3: [
                "저희 {STORE_NM}은 {ORDER_TYPE}이 가능 해요.",
                "{STORE_NM}에서는 {ORDER_TYPE}으로 주문하실 수 있습니다.",
                "저희 {STORE_NM}에서 {ORDER_TYPE} 주문이 가능합니다.",
                "{ORDER_TYPE}을 통해 {STORE_NM}의 서비스를 이용하실 수 있습니다.",
                "저희 {STORE_NM}에서는 {ORDER_TYPE}을 지원합니다."
            ],
            4: [
                "저희 {STORE_NM}에서는 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD}으로 결제 가능해요.",
                "{STORE_NM}에서는 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD} 결제를 지원합니다.",
                "결제는 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD}로 가능하니 참고해주세요.",
                "{STORE_NM}에서는 {PAYMNT_MN_CD}와 {EASY_PAYMNT_TYPE_CD} 방식의 결제가 가능합니다.",
                "결제는 저희 {STORE_NM}에서 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD}로 가능합니다."
            ],
            5: [
                "저희 {STORE_NM}의 영업시간은 {SALE_BGN_TM}부터 {SALE_END_TM}까지에요.",
                "저희 {STORE_NM}은 {SALE_BGN_TM}부터 {SALE_END_TM}까지 운영됩니다.",
                "{STORE_NM}의 영업시간은 {SALE_BGN_TM}에서 {SALE_END_TM}입니다.",
                "운영 시간은 {STORE_NM}에서 {SALE_BGN_TM}부터 {SALE_END_TM}까지입니다.",
                "{STORE_NM}은 매일 {SALE_BGN_TM}부터 {SALE_END_TM}까지 영업합니다."
            ],
            6: [
                "저희 {STORE_NM}의 휴일은 매주 {HOLIDAY_TYPE_CD}이에요.",
                "{STORE_NM}은 {HOLIDAY_TYPE_CD}에 쉬고 있습니다.",
                "매주 {HOLIDAY_TYPE_CD}에 {STORE_NM}은 운영하지 않습니다.",
                "{STORE_NM}의 정기 휴무일은 {HOLIDAY_TYPE_CD}입니다.",
                "{HOLIDAY_TYPE_CD}에 {STORE_NM}은 문을 닫습니다."
            ],
            7: [
                "저희 {STORE_NM}의 배달지역은 {DLVR_ZONE_RADS} 지역에 배달이 가능 합니다.",
                "{STORE_NM}은 {DLVR_ZONE_RADS} 지역까지 배달을 지원합니다.",
                "{DLVR_ZONE_RADS} 지역에 한해 배달이 가능합니다.",
                "{STORE_NM}은 {DLVR_ZONE_RADS}를 포함한 지역에서 배달 가능합니다.",
                "배달 가능 지역은 {DLVR_ZONE_RADS}입니다."
            ],
            8: [
                "저희 {STORE_NM}의 배달 팁은 최소 {DLVR_TIP_AMT_MIN} 부터 최대 {DLVR_TIP_AMT_MAX} 이에요.",
                "{STORE_NM}에서는 배달 팁이 {DLVR_TIP_AMT_MIN}에서 {DLVR_TIP_AMT_MAX}까지 적용됩니다.",
                "배달 팁은 {DLVR_TIP_AMT_MIN} ~ {DLVR_TIP_AMT_MAX} 범위에서 부과됩니다.",
                "{STORE_NM}의 배달 팁은 {DLVR_TIP_AMT_MIN}에서 시작됩니다.",
                "{DLVR_TIP_AMT_MIN}부터 {DLVR_TIP_AMT_MAX}까지 배달 팁이 적용됩니다."
            ],
            9: [
                "저희는 포장 서비스를 {PACK_YN} 있습니다.",
                "포장은 {PACK_YN} 가능합니다.",
                "{STORE_NM}에서는 포장 서비스를 {PACK_YN}고 있습니다.",
                "{PACK_YN} 포장이 제공됩니다."
            ],
            10: [
                "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
                "문의 내용을 확인하고 전화로 답변드리겠습니다.",
                "전화 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
                "문의 확인 후 전화드리겠습니다.",
                "해당 문의는 전화로 추가 안내드리겠습니다."
            ],
            11: [
                "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
                "문의 내용을 확인하고 전화로 답변드리겠습니다.",
                "전화 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
                "문의 확인 후 전화드리겠습니다.",
                "해당 문의는 전화로 추가 안내드리겠습니다."
            ],
            12: [
                "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
                "문의 내용을 확인하고 전화로 답변드리겠습니다.",
                "전화 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
                "문의 확인 후 전화드리겠습니다.",
                "해당 문의는 전화로 추가 안내드리겠습니다."
            ],
            13: [
                "저희 {STORE_NM}의 주소는 {ROAD_NM_ADDR} 에요.",
                "{STORE_NM}의 위치는 {ROAD_NM_ADDR}입니다.",
                "주소: {ROAD_NM_ADDR}. {STORE_NM}입니다.",
                "{ROAD_NM_ADDR}에서 {STORE_NM}을 찾을 수 있습니다.",
                "{STORE_NM}의 정확한 주소는 {ROAD_NM_ADDR}입니다."
            ],
            14: [
                "저희 {STORE_NM}의 테이블 수는 {TABLE_NM}개 에요.",
                "{STORE_NM}의 총 테이블 수는 {TABLE_NM}입니다.",
                "테이블: {TABLE_NM}. {STORE_NM}에서 확인 가능.",
                "{STORE_NM}은 {TABLE_NM}개의 테이블이 있습니다.",
                "{TABLE_NM}개의 테이블을 보유한 {STORE_NM}입니다."
            ],
            15: [
                "저희 {STORE_NM}에는 {ROOM_SEAT}, {OUTDOOR_SEAT}이 각각 {ROOM_TABLE_ZONE_NM}, {OUTDOOR_TABLE_ZONE_NM}개 있어요.",
                "{STORE_NM}은 {ROOM_SEAT} {ROOM_TABLE_ZONE_NM}개와 {OUTDOOR_SEAT} {OUTDOOR_TABLE_ZONE_NM}개를 보유하고 있습니다.",
                "{ROOM_SEAT}, {OUTDOOR_SEAT} 좌석이 {ROOM_TABLE_ZONE_NM}, {OUTDOOR_TABLE_ZONE_NM}개씩 있습니다.",
                "각각 {ROOM_TABLE_ZONE_NM}개의 {ROOM_SEAT}와 {OUTDOOR_TABLE_ZONE_NM}개의 {OUTDOOR_SEAT} 좌석이 준비되어 있습니다.",
                "{STORE_NM}의 좌석 배치는 {ROOM_SEAT} {ROOM_TABLE_ZONE_NM}, {OUTDOOR_SEAT} {OUTDOOR_TABLE_ZONE_NM}입니다."
            ],
            16: [
                "포인트 서비스나 쿠폰을 발급 받기 위해서는 회원 가입이 필요합니다. 회원가입은 스마트 주문 앱 또는 키오스크에서 가입 가능해요.",
                "회원 가입 후 포인트와 쿠폰을 받을 수 있습니다. 스마트 주문 앱을 이용해주세요.",
                "쿠폰과 포인트는 회원 가입 시 제공됩니다. 키오스크에서 가입 가능합니다.",
                "스마트 주문 앱이나 키오스크로 간편하게 회원 가입 후 혜택을 받아보세요.",
                "회원 가입 후 포인트 적립과 쿠폰 발급이 가능합니다."
            ],
            17: [
                "저희 {STORE_NM} 에서는 포인트 {USE_POSBL_BASE_POINT}점 부터 사용 가능해요.",
                "{STORE_NM}의 포인트 사용 기준은 {USE_POSBL_BASE_POINT}점입니다.",
                "{USE_POSBL_BASE_POINT}점 이상부터 포인트 사용 가능합니다.",
                "{STORE_NM}에서 포인트는 {USE_POSBL_BASE_POINT}점부터 이용 가능합니다.",
                "{USE_POSBL_BASE_POINT}점 이상 적립 후 포인트를 사용해보세요."
            ],
            18: [
                "저희 {STORE_NM} 에서는 스탬프 {STAMP_TMS}개 적립 시 {COUPON_PACK_NM} 쿠폰을 발급해 주고 있어요.",
                "{STORE_NM}은 스탬프 {STAMP_TMS}개로 {COUPON_PACK_NM} 쿠폰을 제공합니다.",
                "{STAMP_TMS}개의 스탬프를 모으면 {COUPON_PACK_NM} 쿠폰을 받을 수 있습니다.",
                "스탬프 적립 {STAMP_TMS}개 달성 시 {COUPON_PACK_NM} 쿠폰이 발급됩니다.",
                "{STORE_NM}에서 스탬프 {STAMP_TMS}개로 쿠폰 혜택을 누리세요."
            ],
            19: [
                "현재 저희 {STORE_NM} 에서는 {EVENT_NM} 이벤트가 진행되고 있으며 {EVENT_BNEF_CND_CD} 동안 진행돼요.",
                "{STORE_NM}의 {EVENT_NM} 이벤트는 {EVENT_BNEF_CND_CD}까지입니다.",
                "{EVENT_NM}은 {EVENT_BNEF_CND_CD}까지 {STORE_NM}에서 참여 가능합니다.",
                "저희 {STORE_NM}에서 {EVENT_NM} 이벤트를 진행 중이며, {EVENT_BNEF_CND_CD}까지 진행됩니다.",
                "{EVENT_BNEF_CND_CD}까지 {EVENT_NM} 이벤트를 즐겨보세요."
            ],
            20: [
                "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
                "문의 내용을 확인하고 전화로 답변드리겠습니다.",
                "전화 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
                "문의 확인 후 전화드리겠습니다.",
                "해당 문의는 전화로 추가 안내드리겠습니다."
            ],
            21: [
                "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
                "문의 내용을 확인하고 전화로 답변드리겠습니다.",
                "전화 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
                "문의 확인 후 전화드리겠습니다.",
                "해당 문의는 전화로 추가 안내드리겠습니다."
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
            top_products = df.query("QTY != '없음'")[['STORE_NM', 'STD_PROD_NM', 'QTY']].groupby(['STORE_NM', 'STD_PROD_NM']).max().sort_values(by='QTY', ascending=False).reset_index().head(5)
            result = top_products.groupby('STORE_NM')['STD_PROD_NM'].apply(lambda x: ', '.join(x)).reset_index().rename(columns={'STD_PROD_NM': 'TOT_PROD_QTY'})
        elif intent_id == 2:  # 배지 및 특선 메뉴 안내
            result = df.query("BADGE != '없음' and SPECIAL_PROD != '없음'")[['BADGE', 'SPECIAL_PROD']].drop_duplicates()
        elif intent_id == 3:  # 주문 / 전달 방식 안내
            result = df.query("ORDER_TYPE != '없음' and ORDER_TYPE != ''")[['STORE_NM','ORDER_TYPE']].drop_duplicates()
        elif intent_id == 4:  # 결제 방법 안내
            result = df.query("PAYMNT_MN_CD != '없음' and EASY_PAYMNT_TYPE_CD != '없음'")[['STORE_NM','PAYMNT_MN_CD', 'EASY_PAYMNT_TYPE_CD']].drop_duplicates()
        elif intent_id == 5:  # 영업 시작/종료 시간 안내
            result = df.query("SALE_BGN_TM != '없음' and SALE_END_TM != '없음'")[['STORE_NM','SALE_BGN_TM', 'SALE_END_TM']].drop_duplicates()
        elif intent_id == 6:  # 휴일 정보 안내
            result = df.query("HOLIDAY_TYPE_CD != '없음' and HOLIDAY_TYPE_CD != ''")[['STORE_NM','HOLIDAY_TYPE_CD']].drop_duplicates()
        elif intent_id == 7:  # 배달 가능 지역 안내
            result = df.query("DLVR_ZONE_RADS != '없음' and DLVR_ZONE_RADS != ''")[['STORE_NM','DLVR_ZONE_RADS']].drop_duplicates()
        elif intent_id == 8:  # 배달비 및 최소 주문 금액 안내
            result = df.query("DLVR_TIP_AMT_MIN != '없음' and DLVR_TIP_AMT_MAX != '없음'")[['STORE_NM','DLVR_TIP_AMT_MIN', 'DLVR_TIP_AMT_MAX']].drop_duplicates()
        elif intent_id == 9:  # 포장 서비스 제공 여부 안내
            result = df.query("PACK_YN != '없음'")[['PACK_YN']].drop_duplicates()
        elif intent_id == 13:  # 상점 주소 안내
            result = df.query("ROAD_NM_ADDR != '없음' and ROAD_NM_ADDR != ''")[['STORE_NM', 'ROAD_NM_ADDR']].drop_duplicates()
        elif intent_id == 14:  # 테이블 및 좌석 안내
            result = df.query("TABLE_NM != '없음'")[['STORE_NM','TABLE_NM']].drop_duplicates()
        elif intent_id == 15:  # 룸 및 야외 테이블 안내
            result = df.query("ROOM_TABLE_ZONE_NM != '없음' and OUTDOOR_TABLE_ZONE_NM != '없음'")[['STORE_NM','ROOM_SEAT', 'OUTDOOR_SEAT','ROOM_TABLE_ZONE_NM', 'OUTDOOR_TABLE_ZONE_NM']].drop_duplicates()
        elif intent_id == 17:  # 포인트 사용 기준 안내
            result = df.query("USE_POSBL_BASE_POINT != '없음'")[['STORE_NM','USE_POSBL_BASE_POINT']].drop_duplicates()
        elif intent_id == 18:  # 스탬프 및 쿠폰 안내
            result = df.query("STAMP_TMS != '없음' and COUPON_PACK_NM != '없음'")[['STORE_NM','STAMP_TMS', 'COUPON_PACK_NM']].drop_duplicates()
        elif intent_id == 19:  # 이벤트 및 할인 혜택 안내
            result = df.query("EVENT_NM != '없음' and EVENT_BNEF_CND_CD != '없음'")[['STORE_NM','EVENT_NM', 'EVENT_BNEF_CND_CD']].drop_duplicates()
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
    user_input = "배달 가능한가요?"
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    intent_id = outputs.logits.argmax().item()
    print("인텐트:", intent_id)
    dataframes = load_excel_to_dataframe("/home/user09/beaver/data/shared_files/dataset/dataset_SQL_general_ju_1_105200_preprocessed.xlsx")  
    
    data = execute_sql(intent_id, dataframes)#[list(dataframes.keys())[0]])
    
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