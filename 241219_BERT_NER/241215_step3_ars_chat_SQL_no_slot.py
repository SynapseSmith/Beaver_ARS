import os
import time
import torch
import random
import pandas as pd
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# === Flask App 초기화 === #
app = Flask(__name__)

# === Logger 설정 === #
def load_logger(log_dir, log_level):
    logger = logging.getLogger(__name__)
    if log_level == 'INFO':
        lv = logging.INFO
    elif log_level == 'ERROR':
        lv = logging.ERROR
    elif log_level == 'DEBUG':
        lv = logging.DEBUG
    else:
        raise NotImplementedError
    logger.setLevel(lv)

    formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir, encoding='utf-8-sig')
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


curtime = time.strftime("%Hh-%Mm-%Ss")
date = time.strftime("%Y-%m-%d")
log_folder = os.path.join('/home/user09/beaver/data/shared_files/241215_BERT/logs', date)
if not os.path.exists(log_folder):
    os.mkdir(log_folder)

logdir = os.path.join(log_folder, curtime + '.log')
logger = load_logger(logdir, 'INFO')
logger.info(f'*** {curtime} START ***')
logger.info(f'*** PID: {os.getpid()} ***')

# === Args 클래스 및 모델 로드 === #
class Args:
    def __init__(self):
        self.output_dir = "/home/user09/beaver/data/shared_files/241215_BERT/checkpoint/klue_roberta_large"
        # self.intents_dict = {
        #     0: "메뉴 카테고리 안내",  # STORE_NM, STD_CATEGORY_NM
        #     1: "인기 / 추천 / 신메뉴 안내",  #!!! STORE_NM, INDI_TYPE_NM1, RECOMMEND_MENU(인기, 추천, 대표 뱃지)
        #     2: "계절 한정 메뉴 (예: 여름 특선, 겨울 메뉴) / 프로모션 메뉴 (예: 신메뉴 할인) 안내", # !!!!! STORE_NM, INDI_TYPE_NM3, LIMMIT_MENU(이벤트, 한정메뉴)
        #     3: "주문 / 전달 방식 안내 (키오스크, 테이블오더, 스마트주문 / 내점, 포장, 배달 등)",   # STORE_NM, ORDER_TYPE
        #     4: "결제 방법 안내(현금, 카드, 간편 결제등)",   # STORE_NM, PAYMNT_MN_CD, EASY_PAYMNT_TYPE_CD
        #     5: "영업 시작/종료 시간 안내",   # STORE_NM, SALE_BGN_TM, SALE_END_TM
        #     6: "정기 휴무/주말/공휴일 운영 여부 안내",  # STORE_NM, HOLIDAY_TYPE_CD
        #     7: "배달 가능 지역 안내",   # STORE_NM, DLVR_ZONE_RADS
        #     8: "배달비 및 최소 주문 금액 안내",   # STORE_NM, DLVR_TIP_AMT, ORDER_BGN_AMT
        #     9: "포장 서비스 제공 여부 안내",   # STORE_NM, PACK_YN
        #     10: "예약 가능 여부 안내(전화/온라인)", 
        #     11: "예약 취소/변경 절차 안내",
        #     12: "고객 대기 및 혼잡도 안내",   
        #     13: "상점 주소 안내 및 지도 링크 전달",   # STORE_NM, ROAD_NM_ADDR
        #     14: "테이블 배치 및 좌석 수 안내",   # STORE_NM, TABLE_NM
        #     15: "야외 테라스 또는 개별 룸 여부 안내",
        #     16: "멤버십 가입 / 혜택 안내",
        #     17: "포인트 적립 / 사용 관련 안내",   # STORE_NM, USE_POSBL_BASE_POINT
        #     18: "쿠폰 발행 / 사용 안내",   # STORE_NM, STAMP_TMS, COUPON_PACK_NM
        #     19: "현재 진행 중인 이벤트 및 할인 혜택, 기간 정보 안내",  # STORE_NM, EVENT_NM, EVENT_BNEF_CND_CD
        #     20: "상점 관리자 연결 안내",
        #     21: "fallbackintent"
        # }
        
        self.intents_dict = {
            0: "메뉴 카테고리 안내",
            1: "특정 상품 및 가격 안내", # 슬롯
            2: "상품에 대한 상세 및 추가 안내",
            3: "(뱃지) 인기메뉴",
            4: "(뱃지) 추천메뉴",
            5: "(뱃지) 대표메뉴",
            6: "(뱃지) 할인, 이벤트,",
            48: "(뱃지) 1+1 메뉴 문의",
            7: "(뱃지) 신상 메뉴",
            8: "(뱃지) 한정 메뉴",
            9: "(뱃지) 매운맛",
            10: "주문 방식 안내 (키오스크, 테이블오더, 스마트주문 등)",
            11: "주문한 상품 전달 방식 안내 (내점, 포장, 배달 등)",
            12: "결제 방법 안내(현금, 카드, 간편 결제 등)",
            13: "특정 결제 방법 상세 안내", # 슬롯
            14: "결제 방법 추가 안내",
            15: "영업 시간 안내",
            16: "영업 시간 상세 안내",
            17: "브레이크 타임 안내",
            18: "브레이크 타임 상세 안내",
            19: "휴무일 안내 (정기 - 임시휴무/주말/공휴일)",
            20: "휴무일 상세 안내 (정기 - 임시휴무/주말/공휴일)",
            21: "영업시간 및 휴무일 추가 안내",
            22: "특정 요일에 대한 영업 여부", # 슬롯
            23: "배달 가능 지역 안내",
            24: "배달비 및 최소 주문 금액 안내",
            25: "배달 추가 안내",
            26: "테이블 점유 안내",
            27: "테이블 점유 추가 안내",
            28: "상점 주소 안내 및 지도 링크 전달",
            29: "대중교통 이용 방법 안내",
            30: "근처 랜드마크 안내",
            31: "테이블 및 좌석 수 안내",
            32: "야외 테라스 또는 개별 룸 여부 안내",
            33: "상점 규모 및 시설 추가 안내",
            34: "멤버십 가입 안내",
            35: "멤버십 혜택 안내",
            36: "포인트 적립 안내",
            37: "포인트 사용 안내",
            38: "쿠폰 발행 안내",
            39: "쿠폰 사용 안내",
            40: "멤버십 및 쿠폰에 대한 추가 안내",
            41: "현재 진행 중인 이벤트 안내",
            42: "현재 진행 중인 이벤트 상세 안내",
            43: "현재 진행 중인 이벤트 추가 안내",
            44: "CallBackIntent",
            45: "감사 인텐트",
            46: "인사 인텐트",
            47: "FallBackIntent"
        }
        self.response_templates = {
            0: [
                "저희 {STORE_NM}에서는 {STD_CATEGORY_NM}의 메뉴를 제공합니다. 자세한 사항은 문자로 전송드린 메뉴판을 확인해주세요.",
                "저희 {STORE_NM}에서는 {STD_CATEGORY_NM}의 다양한 메뉴가 준비되어 있습니다.",
                "저희 {STORE_NM}에는 {STD_CATEGORY_NM}와 같은 메뉴들이 있습니다. 자세한 내용은 문자로 전송드린 메뉴판을 참조해주세요.",
                "{STORE_NM}에서는 {STD_CATEGORY_NM}와 같은 메뉴를 판매하고 있습니다."
            ],
            1: "{STD_MENU_NM}의 가격은 {PRICE}입니다. 더 자세한 내용은 문자로 발송해드린 메뉴판을 확인해주세요.",
            2: [
                "저희 {STORE_NM}의 {INDI_TYPE_NM1} 메뉴는 {RECOMMEND_MENU} 메뉴로서 추천드립니다.",
                 "{STORE_NM}에서 {INDI_TYPE_NM1} 메뉴로서 {RECOMMEND_MENU} 메뉴를 추천드립니다.",
                "{STORE_NM}에서 고객님들이 가장 많이 찾는 메뉴는 {RECOMMEND_MENU} 입니다.",
            ],
            3: [
                "저희 매장에서 {INDI_TYPE_NM3} 메뉴는 {LIMMIT_MENU}입니다.",
                "저희 {STORE_NM}에서는 {LIMMIT_MENU} 메뉴를 {INDI_TYPE_NM3} 이벤트중입니다.",
                "지금 {STORE_NM}에서는 {INDI_TYPE_NM3} 이벤트 중인 {LIMMIT_MENU} 메뉴를 만나보실 수 있습니다.",
                "저희 매장에서는 {LIMMIT_MENU} 메뉴를 {INDI_TYPE_NM3} 이벤트로 선보이고 있습니다."
            ],
            4: [
                "저희 {STORE_NM}은 {ORDER_TYPE} 방식으로 주문이 가능해요.",
                "{STORE_NM}에서는 {ORDER_TYPE} 의 방법으로 주문 하실 수 있습니다.",
                "저희 {STORE_NM}에서 {ORDER_TYPE} 방식으로 주문이 가능합니다.",
                "{ORDER_TYPE} 방식을 통해 {STORE_NM}의 서비스를 이용하실 수 있습니다."
            ],
            5: [
                "저희 {STORE_NM}에서는 {PAYMNT_MN_CD} 방식의 결제가 가능하며 간편결제로는 {EASY_PAYMNT_TYPE_CD} 가능합니다.",
                "{STORE_NM}에서는 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD} 방식을 지원합니다.",
                "결제는 {PAYMNT_MN_CD}, {EASY_PAYMNT_TYPE_CD} 방식이 가능하니 참고해주세요.",
                "{STORE_NM}에서는 {PAYMNT_MN_CD} 방식의 결제가 가능합니다.",
            ],
            6: "저희는 {PAYMENT_NM_SPE} 결제가 {PAYMENT_AVAIL} 매장 입니다.",
            7: [
                "저희 {STORE_NM}의 영업시간은 {SALE_BGN_TM}부터 {SALE_END_TM}까지에요.",
                "저희 {STORE_NM}은 {SALE_BGN_TM}부터 {SALE_END_TM}까지 운영됩니다.",
                "{STORE_NM}의 영업시간은 {SALE_BGN_TM}부터 {SALE_END_TM}까지입니다.",
                "운영 시간은 {STORE_NM}에서 {SALE_BGN_TM}부터 {SALE_END_TM}까지입니다.",
                "{STORE_NM}은 {SALE_BGN_TM}부터 {SALE_END_TM}까지 영업합니다."
            ],
            8: "{DAY_NM}에는 영업 {DAY_INFO}고, 입니다.",
            # 8: "{DAY_NM}에는 영업 {DAY_INFO}고, 영업 시간은 {SALE_BGN_TM} 부터 {SALE_END_TM}까지 입니다.",
            9: [
                "저희 {STORE_NM}의 휴일은 {HOLIDAY_TYPE_CD}이에요.",
                "{STORE_NM}은 {HOLIDAY_TYPE_CD}에 휴무일입니다.",
                "{HOLIDAY_TYPE_CD}에 {STORE_NM}은 운영하지 않습니다.",
                "{STORE_NM}의 정기 휴무일은 {HOLIDAY_TYPE_CD}입니다."
            ],
            10: [
                "저희 {STORE_NM}은 반경 {DLVR_ZONE_RADS} 이내의 지역만 배달 가능합니다. 자세한 사항은 매장에 문의해주세요.",
                "{STORE_NM}은 반경 {DLVR_ZONE_RADS} 이내의 지역으로만 배달 서비스를 제공하고 있습니다. 자세한 내용은 매장에 문의해주세요.",
                "현재 {STORE_NM}의 배달 가능 지역은 반경 {DLVR_ZONE_RADS} 이내입니다. 자세한 내용은 매장으로 문의해주세요.",
                "저희 {STORE_NM}의 배달 범위는 반경 {DLVR_ZONE_RADS} 이내 지역입니다. 자세한 사항은 확인 후 안내 드리겠습니다.",
                "{STORE_NM}의 배달 서비스는 매장 반경 {DLVR_ZONE_RADS} 이내에서만 운영됩니다. 해당 부분은 확인 후 다시 안내드리겠습니다."
            ]
            ,
            11: [
                "저희 {STORE_NM}의 배달 팁은 최소 {DLVR_TIP_AMT} 이며 {ORDER_BGN_AMT} 이상 주문해주셔야 배달이 가능합니다.",
                 "저희 {STORE_NM}은 배달 팁이 최소 {DLVR_TIP_AMT}이며, {ORDER_BGN_AMT} 이상 주문 시 배달 가능합니다.",
                "배달 팁은 최소 {DLVR_TIP_AMT}이고, {STORE_NM}에서는 {ORDER_BGN_AMT} 이상 주문해 주셔야 배달이 가능합니다.",
                "저희 {STORE_NM}에서 배달을 원하실 경우 최소 {ORDER_BGN_AMT} 이상 주문 시 배달이 가능하며, 배달 팁은 최소{DLVR_TIP_AMT}입니다.",
                "현재 {STORE_NM}의 배달 팁은 최소 {DLVR_TIP_AMT}이며, 최소 주문 금액은 {ORDER_BGN_AMT}입니다."
            ],
            12: [
                "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요.",
                "문의 내용을 확인하고 문자로 답변드리겠습니다.",
                "문의 확인 후 해당 번호로 문자 안내 드리겠습니다.",
                "해당 문의는 문자로 추가 안내드리겠습니다."
            ],
            13: [
                "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요.",
                "문의 내용을 확인하고 문자로 답변드리겠습니다.",
                "해당 문의에 대해 자세한 확인 후 문자로 답변드리겠습니다.",
                "해당 문의는 문자로 추가 안내드리겠습니다."
            ],
            14: [
                "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요.",
                "문의 내용을 확인하고 문자로 답변드리겠습니다.",
                "매장에 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
                "문의 확인 후 전화드리겠습니다.",
                "해당 문의는 전화로 추가 안내드리겠습니다."
            ],
            15: [
                "저희 {STORE_NM}는 {ROAD_NM_ADDR}에 위치해 있습니다.",
                "{STORE_NM}의 위치는 {ROAD_NM_ADDR}입니다.",
                "{ROAD_NM_ADDR} 주소로 오시면 {STORE_NM} 매장을 찾으실 수 있습니다.",
                "{STORE_NM}의 정확한 주소는 {ROAD_NM_ADDR}입니다."
            ],
            16: [
                "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요.",
                "문의 내용을 확인하고 문자로 답변드리겠습니다.",
                "해당 문의는 매장에 문의 부탁드립니다.",
                "해당 내용은 문의 확인 후 문자 안내드리겠습니다.",
                "해당 정보는 전화번호로 문자 안내드리겠습니다."
            ],
            17: [
                "해당 문의는 확인 후 전화하신 번호로 문자안내 드릴게요.",
                "문의 내용을 확인하고 문자로 답변드리겠습니다.",
                "매장에 문의 부탁드립니다.",
                "해당 내용은 문의 확인 후 문자 안내드리겠습니다.",
                "해당 정보는 전화번호로 문자 안내드리겠습니다."
                # "저희 {STORE_NM}에는 {ROOM_SEAT}, {OUTDOOR_SEAT}이 각각 {ROOM_TABLE_ZONE_NM}, {OUTDOOR_TABLE_ZONE_NM}개 있어요.",
                # "{STORE_NM}은 {ROOM_SEAT} {ROOM_TABLE_ZONE_NM}개와 {OUTDOOR_SEAT} {OUTDOOR_TABLE_ZONE_NM}개를 보유하고 있습니다.",
                # "{ROOM_SEAT}, {OUTDOOR_SEAT} 좌석이 {ROOM_TABLE_ZONE_NM}, {OUTDOOR_TABLE_ZONE_NM}개씩 있습니다.",
                # "각각 {ROOM_TABLE_ZONE_NM}개의 {ROOM_SEAT}와 {OUTDOOR_TABLE_ZONE_NM}개의 {OUTDOOR_SEAT} 좌석이 준비되어 있습니다.",
                # "{STORE_NM}의 좌석 배치는 {ROOM_SEAT} {ROOM_TABLE_ZONE_NM}, {OUTDOOR_SEAT} {OUTDOOR_TABLE_ZONE_NM}입니다."
            ],
            18: [
                "포인트 서비스나 쿠폰을 발급 받기 위해서는 회원 가입이 필요합니다. 회원가입은 스마트 주문 앱 또는 키오스크에서 가입 가능해요.",
                "회원 가입 후 포인트와 쿠폰을 받을 수 있습니다. 스마트 주문 앱을 이용해주세요.",
                "쿠폰과 포인트는 회원 가입 시 제공됩니다. 키오스크에서 가입 가능합니다.",
                "스마트 주문 앱이나 키오스크로 간편하게 회원 가입 후 혜택을 받아보세요.",
                "회원 가입 후 포인트 적립과 쿠폰 발급이 가능합니다."
            ],
            19: [
                "저희 {STORE_NM} 에서는 포인트 {USE_POSBL_BASE_POINT}점 부터 사용 가능해요.",
                "{STORE_NM}의 포인트 사용 기준은 {USE_POSBL_BASE_POINT}점입니다.",
                "{USE_POSBL_BASE_POINT}점 이상부터 포인트 사용 가능합니다.",
                "{STORE_NM}에서 포인트는 {USE_POSBL_BASE_POINT}점부터 이용 가능합니다.",
                "{USE_POSBL_BASE_POINT}점 이상 적립 후 포인트를 사용하실 수 있습니다."
            ],
            20: [
                "저희 {STORE_NM} 에서는 스탬프 {STAMP_TMS}개 적립 시 {COUPON_PACK_NM} 쿠폰을 발급해 주고 있어요.",
                "{STORE_NM}은 스탬프 {STAMP_TMS}개 적립 시 {COUPON_PACK_NM} 쿠폰을 제공합니다.",
                "{STAMP_TMS}개의 스탬프를 모으면 {COUPON_PACK_NM} 쿠폰을 받을 수 있습니다.",
                "스탬프 적립 {STAMP_TMS}개 달성 시 {COUPON_PACK_NM} 쿠폰이 발급됩니다.",
                "{STORE_NM}에서 스탬프 {STAMP_TMS}개 적립을 통해 쿠폰 혜택을 누릴 수 있습니다."
            ],
            21: [
                "현재 저희 {STORE_NM} 에서는 {EVENT_NM} 이벤트가 진행되고 있으며 {EVENT_BNEF_CND_CD} 동안 진행돼요.",
                "{STORE_NM}의 {EVENT_NM} 이벤트는 {EVENT_BNEF_CND_CD}까지입니다.",
                "{EVENT_NM}은 {EVENT_BNEF_CND_CD}까지 {STORE_NM}에서 참여 가능합니다.",
                "저희 {STORE_NM}에서 {EVENT_NM} 이벤트를 진행 중이며, {EVENT_BNEF_CND_CD}까지 진행됩니다.",
                "{EVENT_BNEF_CND_CD}까지 {EVENT_NM} 이벤트를 즐겨보세요."
            ],
            22: [
                "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
                "문의 내용을 확인하고 문자로 답변드리겠습니다.",
                "전화 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
                "문의 확인 후 문자 안내 드리겠습니다.",
                "해당 문의는 문자로 추가 안내드리겠습니다."
            ],
            23: [
                "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요.",
                "문의 내용을 확인하고 문자로 답변드리겠습니다.",
                "매장에 문의 후 안내드리겠습니다. 잠시만 기다려주세요.",
                "문의 확인 후 문자 드리겠습니다.",
                "해당 문의는 문자로 추가 안내드리겠습니다."
            ]
        }



args = Args()

# === 학습된 의도 분류 모델 및 토크나이저 로드 === #
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 데이터베이스 로드 함수 === #
# def load_excel_to_dataframe(file_path):
#     df = pd.read_excel(file_path, sheet_name=0)
#     df.columns = [col.strip().replace(" ", "_") for col in df.columns]
#     df.fillna("없음", inplace=True)
#     return df


def load_excel_to_dataframe(file_path):
    dataframes = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in dataframes.items():
        df.columns = [col.strip().replace(" ", "_") for col in df.columns]
        df.fillna("없음", inplace=True)
    return df

info = {
    '짜장면': '5000원',
    '짬뽕': '7000원',
    '탕수육': '15000원',
    '월요일': '영업하',
    '화요일': '영업하지 않',
    '수요일': '영업하',
    '목요일': '영업하',
    '금요일': '영업하',
    '토요일': '영업하',
    '일요일': '영업하지 않',
    '네이버페이': '가능한',
    '카카오페이': '불가능한',
    '현금': '가능한',
    '신용카드': '가능한'
}

# === SQL-like 쿼리 실행 함수 === #
def execute_sql(intent_id, df):
    try:
        if intent_id == 0:  # 메뉴 카테고리 안내
            result = df.query("STD_CATEGORY_NM != '없음' and STD_CATEGORY_NM != ''")[['STORE_NM', 'STD_CATEGORY_NM']].drop_duplicates()
        elif intent_id == 1:  # 인기 / 추천 메뉴 안내
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM1!= '없음'")[['STORE_NM', 'INDI_TYPE_NM1', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'RECOMMEND_MENU'}).drop_duplicates()
            # result = df.query("QTY != '없음'")[['STORE_NM', 'STD_PROD_NM', 'QTY']].groupby(['STORE_NM', 'STD_PROD_NM']).max().sort_values(by='QTY', ascending=False).reset_index().rename(columns={'STD_PROD_NM': 'TOT_PROD_QTY'})
        elif intent_id == 2:  # 배지 및 특선 메뉴 안내
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM3!= '없음'")[['STORE_NM', 'INDI_TYPE_NM3', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'LIMMIT_MENU'}).drop_duplicates()
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
            result = df.query("DLVR_TIP_AMT != '없음' and ORDER_BGN_AMT != '없음'")[['STORE_NM','DLVR_TIP_AMT', 'ORDER_BGN_AMT']].drop_duplicates()
        # elif intent_id == 9:  # 포장 서비스 제공 여부 안내
        #     result = df.query("PACK_YN != '없음'")[['STORE_NM', 'PACK_YN']].drop_duplicates()
        elif intent_id == 12:  # 상점 주소 안내
            result = df.query("ROAD_NM_ADDR != '없음' and ROAD_NM_ADDR != ''")[['STORE_NM', 'ROAD_NM_ADDR']].drop_duplicates()
        # elif intent_id == 13:  # 테이블 및 좌석 안내
        #     result = df.query("TABLE_NM != '없음'")[['STORE_NM','TABLE_NM']].drop_duplicates()
        # elif intent_id == 14:  # 룸 및 야외 테이블 안내
        #     result = df.query("ROOM_TABLE_ZONE_NM != '없음' and OUTDOOR_TABLE_ZONE_NM != '없음'")[['STORE_NM','ROOM_SEAT', 'OUTDOOR_SEAT','ROOM_TABLE_ZONE_NM', 'OUTDOOR_TABLE_ZONE_NM']].drop_duplicates()
        elif intent_id == 16:  # 포인트 사용 기준 안내
            result = df.query("USE_POSBL_BASE_POINT != '없음'")[['STORE_NM','USE_POSBL_BASE_POINT']].drop_duplicates()
        elif intent_id == 17:  # 스탬프 및 쿠폰 안내
            result = df.query("STAMP_TMS != '없음' and COUPON_PACK_NM != '없음'")[['STORE_NM','STAMP_TMS', 'COUPON_PACK_NM']].drop_duplicates()
        elif intent_id == 18:  # 이벤트 및 할인 혜택 안내
            result = df.sample(frac=1, random_state=None).query("EVENT_NM != '없음' and EVENT_BNEF_CND_CD != '없음'")[['STORE_NM','EVENT_NM', 'EVENT_BNEF_CND_CD']].drop_duplicates()
        elif intent_id in [9, 10, 11, 13, 14, 15, 19, 20]:  # 고정 응답이 필요한 intent_id
            print(f"Static response required for intent_id {intent_id}")
            return None
        else:
            print("해당 intent에 대한 처리가 없습니다.")
            return None
        return result.iloc[0].to_dict() if not result.empty else None
    except Exception as e:
        print(f"쿼리 실행 오류: {e}")
        return None
    

# === 의도 분류 함수 === #
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return predicted_class

# === 응답 생성 함수 === #
def generate_response(intent_id, data):
    templates = args.response_templates.get(intent_id, "죄송합니다, 요청을 처리할 수 없습니다.")
    if isinstance(templates, list):
        template = random.choice(templates)  # 리스트에서 랜덤하게 템플릿 선택
    else:
        template = templates
    
    try:
        return template.format(**data)
    except KeyError as e:
        logger.error(f"응답 템플릿에 필요한 데이터가 누락되었습니다: {e}")
        return "필요한 데이터가 부족하여 응답할 수 없습니다."

# === Flask 엔드포인트 === #
@app.route('/order', methods=['POST'])
def order():
    start_time = time.time()
    try:
        # 요청에서 메시지 추출
        event = request.get_json(force=True)
        user_message = event['body'].get('text', "")
        logger.info(f"Received message: {user_message}")

        # 의도 분류
        intent_id = classify_intent(user_message)
        logger.info(f"Intent ID: {intent_id}")

        # 데이터 로드 및 SQL-like 실행
        data_path = "/home/user09/beaver/data/shared_files/dataset/dataset_SQL_general_ju_2_hong_preprocessed.xlsx"
        df = load_excel_to_dataframe(data_path)
        result_data = execute_sql(intent_id, df)
        print("result_data:",result_data)

        # 응답 생성
        response_text = generate_response(intent_id, result_data) if result_data else random.choice(args.response_templates[intent_id])
        logger.info(f"Response: {response_text}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Processing Time: {processing_time:.4f} seconds")
        logger.info('-'*100)
        
        return jsonify({"response": response_text, "processing_time": f"{processing_time:.4f} seconds"})

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"response": "오류가 발생했습니다. 다시 시도해 주세요."})

# === Flask 앱 실행 === #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5060, debug=False)
