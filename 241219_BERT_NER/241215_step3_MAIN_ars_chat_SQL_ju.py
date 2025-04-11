import os
import time
import torch
import random
import pandas as pd
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from rank_bm25 import BM25Okapi
import re
from transformers import AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
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
log_folder = os.path.join('/home/user09/beaver/data/shared_files/241219_BERT_NER/logs', date)
if not os.path.exists(log_folder):
    os.mkdir(log_folder)

logdir = os.path.join(log_folder, curtime + '.log')
logger = load_logger(logdir, 'INFO')
logger.info(f'*** {curtime} START ***')
logger.info(f'*** PID: {os.getpid()} ***')

# === Args 클래스 및 모델 로드 === #
class Args:
    def __init__(self):
        self.label_list = ["O", "B-MENU", "I-MENU", "B-PAYMENT", "I-PAYMENT", "B-DAY"]
        self.model_checkpoint_path = "/home/user09/beaver/data/shared_files/241218_NER/ner_checkpoint2"
        self.sim_threshold = 0.5
        self.output_dir = "/home/user09/beaver/data/shared_files/241219_BERT_NER/checkpoint/klue_roberta_large_v6"
        self.data_path = "/home/user09/beaver/data/shared_files/241219_BERT_NER/data/dataset_SQL_general_ju_5_hong_preprocessed.xlsx"
        
        self.intent_dict = {
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
        self.response_templates = {
            0: [
                "저희 매장에는 {STD_CATEGORY_NM}의 메뉴 종류들이 있어요."
            ],
            1: [
                "{STD_MENU_NM}의 가격은 {PRICE}원 입니다. 더 자세한 내용은 문자로 발송해드린 메뉴판을 확인해주세요."
            ],
            2: [
                "자세한 메뉴 사진과 정보가 적혀있는 QR 주문링크를 전송해드렸어요. 스마트폰으로 편하게 확인해 보세요!"
            ],
            3: [
                "저희 매장에서는 {POPULAR_MENU} 메뉴가 {INDI_TYPE_NM5} 메뉴에요."
            ],
            4: [
                "저희 매장에서는 {RECOMMEND_MENU} 메뉴를 {INDI_TYPE_NM1} 해요."
            ],
            5: [
                "저희 매장에서는 {REPRE_MENU} 메뉴가 {INDI_TYPE_NM4} 메뉴에요."
            ],
            6: [  # 할인, 이벤트
                "저희 매장에서 할인하는 이벤트 메뉴는 {EVENT_MENU} 메뉴에요."
            ],
            7: [  # 1+1
                "저희 매장에서 {INDI_TYPE_NM7} 메뉴는 {PLUS_MENU} 메뉴를 제공하고 있어요."
                ],
            8: [
                "저희 매장에서 {NEW_MENU} 메뉴는 {INDI_TYPE_NM8} 입니다."
            ],
            9: [
                "저희 매장에서는 {INDI_TYPE_NM6} 메뉴는 {LIMIT_MENU}입니다."
            ],
            10: [
                "저희 매장에서는 {SPICY_MENU} 메뉴가 {INDI_TYPE_NM2} 메뉴입니다."
            ],
            11: [
                "저희 매장에서는 {ORDER_CHNL_CD}로 주문 가능하세요."
            ],
            12: [
                "저희 매장에서는 {ORDER_TYPE} 방식으로 주문이 가능해요."
            ],
            13: [
                "저희 매장에서는 {PAYMNT_MN_CD} 방식의 결제가 가능하며 간편결제로는 {EASY_PAYMNT_TYPE_CD} 방식이 가능해요."
            ],
            14: [
                "저희 매장에서는 {PAYMENT_NM_SPE} 결제 방식으로 {PAYMENT_AVAIL}"
            ],
            15: [
                "결제에 관한 구체적인 문의는 확인 후 전화하신 번호로 다시 연락드릴게요."
            ],
            16: [
                "저희 매장의 영업시간은 {SALE_BGN_TM}에 문을 열고 {SALE_END_TM}에 영업을 마감해요. 휴무일이나 특정 요일 영업시간이 궁금하시면 편하게 물어봐 주세요."
            ],
            17: [
                "저희 매장의 브레이크타임은 {STORE_REST_BGN_TM}에 시작해서 {STORE_REST_END_TM}에 끝난 후 다시 영업을 시작합니다."
            ],
            18: [
                "저희 매장의 휴일은 {HOLIDAY_TYPE_CD}이에요. 영업일은 월요일 또는 수요일에서 일요일입니다."
            ],
            19: [
                "네, 운영합니다. 영업시간은 {SALE_BGN_TM}부터 {SALE_END_TM} 까지에요."
            ],
            20: [
                "영업시간에 대한 구체적인 문의는 확인 후 전화하신 번호로 다시 연락드릴게요."
            ],
            21: [
                "저희 매장은 {DAY_NM}에 영업 {DAY_INFO}해요. {SALE_BGN_TM}부터 {SALE_END_TM} 사이에 편하게 방문해주세요."
            ],
            22: [
                "저희 매장으로부터 반경 {DLVR_ZONE_RADS} 이내 지역에서 배달이 가능해요."
                ],
            
            23: [
                "저희 매장의 배달 팁은 최소 {DLVR_TIP_AMT}이며, 최소 주문 금액은 {ORDER_BGN_AMT}이에요."],
            24: [
                "배달에 대한 구체적인 문의는 확인 후 전화하신 번호로 다시 연락드릴게요."],
            25: [
                "지금 바로 이용하실 수 있어요. {ROOM_TABLE_ZONE_AVAIL}개 비어있습니다.",  # ---> if문!! "조금 기다리셔야 해요. 테이블 {ROOM_TABLE_ZONE_ENTIRE}개 모두 이용중이에요."
                 ],
            26: [
                "테이블 이용에 대한 구체적인 문의는 확인 후 전화하신 번호로 다시 연락드릴게요."
                ],
            27: [
                "저희 매장의 주소는 {ROAD_NM_ADDR} 입니다. 해당 문의를 돕고자 지도 URL 정보를 전송해드릴게요."
                ],
            28: [
                "지도에서 보시는게 더 정확하실 것 같아서 지도 링크를 전송해드렸어요."
                ],
            29: [
                "지도에서 보시는게 더 정확하실 것 같아서 지도 링크를 전송해드렸어요."
                ],
            30: [
                "저희 매장은 {TABLE_NM}개의 테이블을 보유하고 있으며 대략 {MAX_PEOPLE}명까지 수용 가능해요."
                ],
            31: [
                "저희 매장에는 단체석 및 예약석이 {TEAM_SEAT_YN}습니다. 예약을 원하시면 매장으로 문의 부탁드려요."
                ],
            32: [
                "저희 매장에는 특별 좌석으로 룸 좌석 {ROOM_TABLE_ZONE_ENTIRE}개와 야외 좌석 {OUTDOOR_TABLE_ZONE_ENTIRE}개를 제공하고 있어요."
                 ],
            33: [
                "좌석 이용에 대한 구체적인 문의는 확인 후 전화하신 번호로 다시 연락드릴게요."
                 ],
            34: [
                "저희 매장에는 멤버십 서비스가 있어요. 포인트 서비스나 쿠폰을 발급 받기 위해서는 스마트 주문 앱 또는 키오스크에서 가입이 필요해요."
                ],
            35: [
                "포인트 서비스나 쿠폰을 발급 받을 수 있어요."
                ],
            36: [
                "주문 금액에 {POINT_BASE_ACCML_RATE}퍼센트 만큼 적립 가능해요."
                ],
            37: [
                "저희 매장에서는 포인트 {USE_POSBL_BASE_POINT}점 부터 사용 가능해요."
                ],
            38: [
                "저희 매장에서는 스템프 {STAMP_TMS}개 적립시 {COUPON_PACK_NM}을 발급해 주고 있어요."
                ],
            39: [
                "저희 매장에서는 {USE_MIN_ORDER_AMT}원 이상 구매 시 쿠폰 사용이 가능합니다. 쿠폰은 현금 대신 사용하실 수 있어요."
                ],
            40: [
                "멤버십, 포인트 및 쿠폰 관련한 자세한 문의는 확인 후 전화하신 번호로 안내 드릴게요."
                ],
            41: [
                "현재 저희 매장에서는 {EVENT_NM} 이벤트가 진행되고 있어요."
                ],
            42: [
                "진행중인 이벤트에 대한 자세한 정보는 스마트주문 앱에서 확인하실 수 있어요. 전화주신 번호로 스마트주문 설치 링크를 보내드렸어요."
                ], 
            43: [
                "이벤트에 대한 구체적인 문의는 확인 후 전화하신 번호로 다시 연락드릴게요."
                ],
            44: [
                "해당 문의는 확인 후 전화하신 번호로 안내 드릴게요."
                ],
            45: [
                "이용해주셔서 감사합니다. 추가적인 문의사항 있으시면 또 문의해주세요."
                ],
            46: [
                "안녕하세요 홍콩반점입니다. 문의사항 있으시면 말씀해주세요." ## 상점명 들어갈 예정
                ],
            47: [
                "현재 제가 말씀드리기는 어려운 요청이에요."
                ]
            }
        



args = Args()

# === 학습된 의도 분류 모델 및 토크나이저 로드 === #
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
embedding_model = SentenceTransformer('nlpai-lab/KoE5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델과 토크나이저 로드
NER_tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint_path)
NER_model = AutoModelForTokenClassification.from_pretrained(args.model_checkpoint_path, num_labels=len(args.label_list))
NER_model.to("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능 시 GPU로 로드

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

def data_normalization(target_entity):
    target_entity = str(target_entity).replace(' ', '')
    
    return target_entity

# === SQL-like 쿼리 실행 함수 === #
def execute_sql(intent_id, df):
    try:
        if intent_id == 0:  # 메뉴 카테고리 안내
            result = df.query("STD_CATEGORY_NM != '없음' and STD_CATEGORY_NM != ''")[['STD_CATEGORY_NM']].drop_duplicates()
        # elif intent_id == 1:  # 인기 / 추천 메뉴 안내
        #     result = df.sample(frac=1, random_state=None).query("STD_MENU_NM!= '없음'")[['STD_MENU_NM', 'PRICE']].rename(columns={'STD_PROD_NM': 'SPE_MENU_NM'}).drop_duplicates()
        # 2번 --> 단순 탬플릿만으로 처리 가능
        elif intent_id == 3:    
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM5!= '없음'")[['INDI_TYPE_NM5', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'POPULAR_MENU'}).drop_duplicates()
        elif intent_id == 4:    
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM1!= '없음'")[['INDI_TYPE_NM1', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'RECOMMEND_MENU'}).drop_duplicates()
        elif intent_id == 5:    
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM4!= '없음'")[['INDI_TYPE_NM4', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'REPRE_MENU'}).drop_duplicates()
        elif intent_id == 6:  
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM3!= '없음'")[['STORE_NM', 'INDI_TYPE_NM3', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'EVENT_MENU'}).drop_duplicates()
        elif intent_id == 7:  
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM7!= '없음'")[['INDI_TYPE_NM7', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'PLUS_MENU'}).drop_duplicates()
        elif intent_id == 8:  
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM8!= '없음'")[['INDI_TYPE_NM8', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'NEW_MENU'}).drop_duplicates()
        elif intent_id == 9:  
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM6!= '없음'")[['INDI_TYPE_NM6', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'LIMIT_MENU'}).drop_duplicates()
        elif intent_id == 10:  
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM2!= '없음'")[['INDI_TYPE_NM2', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'SPICY_MENU'}).drop_duplicates()
        elif intent_id == 11:  
            result = df.sample(frac=1, random_state=None).query("ORDER_CHNL_CD!= '없음'")[['ORDER_CHNL_CD']].drop_duplicates()
        elif intent_id == 12:  
            result = df.sample(frac=1, random_state=None).query("ORDER_TYPE!= '없음'")[['ORDER_TYPE']].drop_duplicates()
        elif intent_id == 13: 
            result = df.sample(frac=1, random_state=None).query("PAYMNT_MN_CD!= '없음' and EASY_PAYMNT_TYPE_CD != '없음'")[['PAYMNT_MN_CD', 'EASY_PAYMNT_TYPE_CD']].drop_duplicates()
        # elif intent_id == 13: 
        #     result = df.sample(frac=1, random_state=None).query("PAYMENT_NM_SPE!= '없음', PAYMENT_AVAIL!= '없음'")[['PAYMENT_NM_SPE', 'PAYMENT_AVAIL']].drop_duplicates()
        elif intent_id == 16: 
            result = df.query("SALE_BGN_TM!= '없음' and SALE_END_TM!= '없음'")[['SALE_BGN_TM', 'SALE_END_TM']].drop_duplicates()
            # result = df.sample(frac=1, random_state=None).query("SALE_BGN_YM!= '없음', SALE_END_TM!= '없음'")[['SALE_BGN_TM', 'SALE_END_TM']].drop_duplicates()
        elif intent_id == 17: 
            result = df.sample(frac=1, random_state=None).query("STORE_REST_BGN_TM!= '없음' and STORE_REST_END_TM!= '없음'")[['STORE_REST_BGN_TM', 'STORE_REST_END_TM']].drop_duplicates()
        elif intent_id == 18: 
            result = df.sample(frac=1, random_state=None).query("HOLIDAY_TYPE_CD!= '없음'")[['HOLIDAY_TYPE_CD']].drop_duplicates()
        elif intent_id == 19: 
            result = df.sample(frac=1, random_state=None).query("SALE_BGN_TM!= '없음' and SALE_END_TM!= '없음'")[['SALE_BGN_TM', 'SALE_END_TM']].drop_duplicates()
        # elif intent_id == 21: 
        #     result = df.sample(frac=1, random_state=None).query("DAY_NM!= '없음', DAY_INFO!= '없음'")[['DAY_NM', 'DAY_INFO', 'SALE_BGN_TM', 'SALE_END_TM']].drop_duplicates()
        elif intent_id == 22:
            result = df.query("DLVR_ZONE_RADS != '없음' and DLVR_ZONE_RADS != ''")[['DLVR_ZONE_RADS']].drop_duplicates()
        elif intent_id == 23:
            result = df.query("DLVR_TIP_AMT != '없음' and ORDER_BGN_AMT != '없음'")[['DLVR_TIP_AMT', 'ORDER_BGN_AMT']].drop_duplicates()
        elif intent_id == 25:
            result = df.sample(frac=1, random_state=None).query("ROOM_TABLE_ZONE_AVAIL != '없음' and ROOM_TABLE_ZONE_AVAIL != ''")[['ROOM_TABLE_ZONE_AVAIL']].drop_duplicates()
        elif intent_id == 27:
            result = df.query("ROAD_NM_ADDR != '없음' and ROAD_NM_ADDR != ''")[['ROAD_NM_ADDR']].drop_duplicates()
        elif intent_id == 30:
            result = df.query("TABLE_NM != '없음' and TABLE_NM != ''")[['TABLE_NM']].drop_duplicates().assign(MAX_PEOPLE=lambda x: x['TABLE_NM'].astype(int) * 4)[['TABLE_NM', 'MAX_PEOPLE']]
        elif intent_id == 31: ### 단체석 및 예약석 유무 안내 작성 필요요
            result = df.query("TEAM_SEAT_YN != '없음' and TEAM_SEAT_YN != ''")[['TEAM_SEAT_YN']].drop_duplicates()
        elif intent_id == 32:   # ROOM_TABLE_ZONE_ENTIRE, OUTDOOR_TABLE_ZONE_ENTIRE
            result = df.query("ROOM_TABLE_ZONE_ENTIRE != '없음' and OUTDOOR_TABLE_ZONE_ENTIRE != '없음'")[['ROOM_TABLE_ZONE_ENTIRE', 'OUTDOOR_TABLE_ZONE_ENTIRE']].drop_duplicates()
        elif intent_id == 36:   # POINT_BASE_ACCML_RATE
            result = df.query("POINT_BASE_ACCML_RATE != '없음' and POINT_BASE_ACCML_RATE != ''")[[ 'POINT_BASE_ACCML_RATE']].drop_duplicates()
        elif intent_id == 37:
            result = df.query("USE_POSBL_BASE_POINT != '없음'")[['USE_POSBL_BASE_POINT']].drop_duplicates()
        elif intent_id == 38:
            result = df.query("STAMP_TMS != '없음' and COUPON_PACK_NM != '없음'")[['STORE_NM','STAMP_TMS', 'COUPON_PACK_NM']].drop_duplicates()
        elif intent_id == 39:   # USE_MIN_ORDER_AMT
            result = df.query("USE_MIN_ORDER_AMT != '없음' and USE_MIN_ORDER_AMT != ''")[['USE_MIN_ORDER_AMT']].drop_duplicates()
        elif intent_id == 41:   # EVENT_NM 1개
            result = df.query("EVENT_NM != '없음' and EVENT_NM != ''").sample(n=3, random_state=None)
            
        elif intent_id in [2, 15, 20, 24, 26, 28, 29, 33, 34, 35, 40, 42, 43, 44, 45, 46, 47]:  # 고정 응답이 필요한 intent_id
            print(f"Static response required for intent_id {intent_id}")
            return None
        
        elif intent_id == [1, 14, 21]: # 슬롯 채울 넘버들
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

def initialize_embeddings(info_dict):
    keys = list(info_dict.keys())
    embeddings = embedding_model.encode(keys)  # 메뉴 키를 임베딩
    return keys, embeddings

# 유사도 계산 함수
def get_best_match(user_input, info_dict, cached_embeddings, cached_keys, yn=False):
    
    # 사용자 입력 임베딩
    user_embedding = embedding_model.encode([user_input])[0]

    # 코사인 유사도 계산
    cosine_sim = util.cos_sim(user_embedding, cached_embeddings).squeeze().tolist()
    best_index = cosine_sim.index(max(cosine_sim))
    best_match = cached_keys[best_index]
    best_value = list(info_dict.values())[best_index]
    best_score = max(cosine_sim)
    print(best_match, best_score)
    
    if yn:
        if best_score < args.sim_threshold:
            available = "불가능해요."
        else:
            available = "가능해요."
        return best_match, best_value, best_score, available
    return best_match, best_value, best_score

menu_info = {
    "쟁반짜장": "16000",
    "고기짬뽕": "9500",
    "고기짜장": "8500",
    "고추짬뽕": "8900",
    "고추짜장": "8000",
    "짬뽕": "7800",
    "짜장": "6500",
    "탕짬면": "10500",
    "탕짜면": "10500",
    "백짬뽕": "8900",
    "냉짬뽕": "8900",
    "볶음짬뽕": "9000",
    "15주년행사 짜장면": "3900",
    "15주년행사 짬뽕": "5900",
    "굴짬뽕": "8900",
    "행사 탕짜면": "8000",
    "탕짬밥": "10500",
    "탕짜밥": "10500",
    "고기짬뽕밥": "9500",
    "고추짬뽕밥": "8900",
    "짬뽕밥": "7800",
    "짜장밥": "8500",
    "15주년행사 짜장밥": "7500",
    "15주년행사 짬뽕밥": "6800",
    "15주년행사 고추짬뽕밥": "7900",
    "15주년행사 고기짬뽕밥": "8500",
    "굴짬뽕 밥": "8900",
    "백짬뽕 밥": "8900",
    "계란후라이": "900",
    "공기밥": "1000",
    "해쉬브라운": "1000",
    "만능매운소스": "800",
    "만능매운소스 5개": "3500",
    "양념소스": "500",
    "짬뽕국물": "1000",
    "3천원할인 식사2인 탕수육": "26800",
    "2인세트": "29800",
    "2천원할인 쟁반짜장 탕수육": "30800",
    "2천원할인 식사2인 탕수육": "27800",
    "4인세트": "47800",
    "3인세트": "35300",
    "1인세트": "9000",
    "식사 4인세트": "44800",
    "깐풍새우": "19900",
    "깐풍기": "18900",
    "꿔바러우": "18900",
    "쟁반볶음짬뽕": "17000",
    "크림새우": "19900",
    "초록매실": "700",
    "환타": "2000",
    "사이다": "2000",
    "제로콜라": "2000",
    "펩시콜라": "2000",
    "스프라이트": "2000",
    "코카콜라": "2000",
    "제로스프라이트": "2000",
    "뽀로로": "2000",
    "암바사": "2000",
    "연태고량주": "24000",
    "켈리": "5000",
    "카스": "5000",
    "대선": "5000",
    "새로": "5000",
    "좋은데이": "5000",
    "처음처럼": "5000",
    "참이슬 오리지널": "5000",
    "진로": "5000",
    "국산소주": "5000",
    "국산맥주": "5000",
    "칭따오": "7500",
    "테라": "5000",
    "홍콩 하이볼": "5900",
    "미상 하이볼": "5900",
    "니하오": "17000",
    "산지천": "18000",
    "행사생맥주": "1900",
    "5천원할인 연태구냥 중": "19000",
    "산지천 반병": "9900",
    "5천원할인 산지천": "13000",
    "단팥춘권": "3000",
    "멘보샤": "9900",
    "연유꽃빵": "3000",
    "군만두": "6000",
    "군만두 반접시": "3500",
    "탕수육": "18800",
    "탕수육 행사": "9900",
    "해물육교자": "4000"
}

payment_info = {
    '신용카드': '',
    '현금': '',
    '모바일상품권': '',
    '현금': '',
    '간편결제': '',
    '네이버페이': '',
    '카드': '',
}

holiday_info = {
    '월요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},  # !!!!!!!!!!! 안하는 경우에는 영업 시작 시간 부분에 영업하는 요일 붙이기!!
    '화요일': {'영업여부': '안', '영업시작시간': '월요일 또는 수요일에서 일요일 11시', '영업종료시간': '20시 30분'},
    '수요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '목요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '금요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '토요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '일요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'}
}

# 초기화 시 메뉴 임베딩 생성
menu_keys, menu_embeddings = initialize_embeddings(menu_info)
payment_keys, payment_embeddings = initialize_embeddings(payment_info)
holiday_keys, holiday_embeddings = initialize_embeddings(holiday_info)

# NER 인퍼런스 함수
def get_named_entities(text, model, tokenizer, label_list):
    # 토큰화
    tokens = text.strip().split()
    inputs = NER_tokenizer(tokens, return_tensors="pt", truncation=True, is_split_into_words=True)
    word_ids = inputs.word_ids()
    inputs = {k: v.to(NER_model.device) for k, v in inputs.items()}  # 모델 디바이스로 이동
    
    # 모델 추론
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=2).squeeze().tolist()
    

    id_to_label = {i: l for i, l in enumerate(label_list)}
    results = []
    for token_id, pred_label_id in zip(word_ids, predictions):
        if token_id is not None:
            label = id_to_label[pred_label_id]
            token = tokens[token_id]
            results.append((token, label))
    
    # B-, I- 태그를 가진 엔티티만 추출
    entities = []
    current_entity = []
    current_label = None
    for token, label in results:
        if label.startswith("B-"):
            # 새로운 엔티티 시작
            if current_entity:
                # 이전 엔티티 저장
                entities.append((current_label, " ".join(current_entity)))
            current_entity = [token]
            current_label = label[2:]  # B-MENU -> MENU
        elif label.startswith("I-") and current_entity:
            # 현재 엔티티 연장
            curr_label = label[2:]
            if curr_label == current_label:  # 동일한 엔티티 라벨일 경우 연장
                current_entity.append(token)
            else:
                # 라벨이 바뀌었으면 이전 엔티티 종료 후 새로운 엔티티 시작
                entities.append((current_label, " ".join(current_entity)))
                current_entity = [token]
                current_label = curr_label
        else:
            # O 라벨 또는 이전 엔티티 종료
            if current_entity:
                entities.append((current_label, " ".join(current_entity)))
                current_entity = []
                current_label = None
    
    # 마지막 엔티티가 있다면 저장
    if current_entity:
        entities.append((current_label, " ".join(current_entity)))

    return entities

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
        data_path = args.data_path
        df = load_excel_to_dataframe(data_path)
        result_data = execute_sql(intent_id, df)
        print("result_data:",result_data)

        if intent_id in [1, 14, 21]:
            # NER 먼저 수행
            ner_entities = get_named_entities(user_message, NER_model, NER_tokenizer, args.label_list)
            print("엔티티 인식:", ner_entities)
            # ner_entities 예: [("MENU", "짜장면"), ("DAY", "월요일"), ("PAYMENT", "네이버페이") ...]
            
            # NER 결과 중 intent_id에 해당하는 엔티티 추출
            # intent_id == 1: MENU 관련
            # intent_id == 6: PAYMENT 관련
            # intent_id == 8: DAY 관련
            target_entity = None
            if intent_id == 1:
                menu_entities = [ent_text for ent_label, ent_text in ner_entities if ent_label == "MENU"]
                if menu_entities:
                    target_entity = data_normalization(" ".join(menu_entities))

            elif intent_id == 14:
                payment_entities = [ent_text for ent_label, ent_text in ner_entities if ent_label == "PAYMENT"]
                if payment_entities:
                    target_entity = data_normalization(" ".join(payment_entities))

            elif intent_id == 21:
                day_entities = [ent_text for ent_label, ent_text in ner_entities if ent_label == "DAY"]
                if day_entities:
                    target_entity = data_normalization(" ".join(day_entities))
            
                    
        if intent_id in [1, 14, 21]:
            if intent_id == 1:
                cached_keys, cached_embeddings = menu_keys, menu_embeddings
                select_info = menu_info
                
            elif intent_id == 14:
                cached_keys, cached_embeddings = payment_keys, payment_embeddings
                select_info = payment_info
                
            elif intent_id == 21:
                cached_keys, cached_embeddings = holiday_keys, holiday_embeddings
                select_info = holiday_info
            
            extracted_info = get_best_match(target_entity or user_message, select_info, cached_embeddings, cached_keys, yn=intent_id == 14)
            if extracted_info:
                if intent_id == 14:
                    best_match, best_value, best_score, available = extracted_info
                else:
                    best_match, best_value, best_score = extracted_info
                
                if intent_id == 1 and best_score < args.sim_threshold:
                    response_text = "해당 메뉴는 판매하고 있지 않습니다. 다른 메뉴를 선택해주세요."
                    logger.info(f"Response: {response_text}")
                    return jsonify({"response": response_text, "processing_time": f"{time.time() - start_time:.4f} seconds"})        
            ## info 딕셔너리에서 사용자 입력(user_message)와 가장 유사한 키를 찾음
            #if intent_id == 14:  # 가능 여부 판단을 위해서 임계값 이상인 게 있는 경우는 있다고 판단, 없는 경우 없다고 판단.
            #   extracted_info = get_best_match(target_entity, select_info, yn=True)
            #    
            #    if extracted_info:
            #        best_match, best_value, best_score, available = extracted_info  # 언패킹
            #        
            #       print(best_score)
            #else:
            #    extracted_info = get_best_match(user_message, select_info)
            #    if extracted_info:
            #        best_match, best_value, best_score = extracted_info # 언패킹
                    
                 
            # SQL 결과 가져오기 (execute_sql 호출)
            result_data = execute_sql(intent_id, df)
            if result_data is None:
                result_data = {}

            # info 딕셔너리에서 가져온 key-value와 SQL에서 가져온 result_data를 합쳐 최종 데이터 구성
            if intent_id == 1:
                # 특정 메뉴 안내: {STD_MENU_NM}, {PRICE} 외에 필요시 result_data를 확장 가능
                final_data = {"STD_MENU_NM": best_match, "PRICE": best_value}
                final_data.update(result_data)  # SQL 결과 합치기
                response_text = generate_response(intent_id, final_data)
                print(f"매칭된 메뉴: {best_match}, 가격: {best_value}, 유사도 점수: {best_score}")
                print(final_data)

            elif intent_id == 14:
                # print("타겟엔티티:", ner_entities[0][1])
                # 특정 결제 수단 안내: {PAYMENT_NM_SPE}, {PAYMENT_AVAIL}
                final_data = {"PAYMENT_NM_SPE": best_match, "PAYMENT_AVAIL": available}
                final_data.update(result_data)  # SQL 결과 합치기
                response_text = generate_response(intent_id, final_data)

            elif intent_id == 21:
                # 특정 요일 영업 시작/종료 시간 안내: {DAY_NM}, {DAY_INFO} + SQL결과의 {SALE_BGN_TM}, {SALE_END_TM}
                final_data = {"DAY_NM": best_match, "DAY_INFO": best_value['영업여부'], "SALE_BGN_TM": best_value['영업시작시간'], "SALE_END_TM": best_value['영업종료시간']}
                print("final_data:", final_data)
                final_data.update(result_data)  # SQL 결과 합치기
                response_text = generate_response(intent_id, final_data)
                    
        else: 
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
    app.run(host='0.0.0.0', port=5050, debug=False)
