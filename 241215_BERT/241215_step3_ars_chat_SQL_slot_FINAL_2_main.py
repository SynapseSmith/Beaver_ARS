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
        self.label_list = ["O", "B-MENU", "I-MENU", "B-PAYMENT", "I-PAYMENT", "B-DAY"]
        self.model_checkpoint_path = "/home/user09/beaver/data/shared_files/241218_NER/ner_checkpoint"
        self.sim_threshold = 0.3
        self.output_dir = "/home/user09/beaver/data/shared_files/241215_BERT/checkpoint/klue_roberta_large_v3"
        self.data_path = "/home/user09/beaver/data/shared_files/241215_BERT/data/dataset_SQL_general_ju_2_hong_preprocessed.xlsx"
        
        self.intents_dict = {
            0: "메뉴 카테고리 안내",
            1: "특정 메뉴 안내",
            2: "인기 / 추천",
            3: "계절 한정 메뉴 (예: 여름 특선, 겨울 메뉴) / 프로모션 메뉴 (예: 신메뉴 할인) 안내 / 신메뉴",
            4: "주문 / 전달 방식 안내 (키오스크, 테이블오더, 스마트주문 / 내점, 포장, 배달 등)",
            5: "결제 방법 안내(현금, 카드, 간편 결제등)",
            6: "특정 결제 수단 안내",
            7: "영업 시작/종료 시간 안내",
            8: "특정 요일 영업 시작/종료 시간 안내",
            9: "정기 휴무/주말/공휴일 운영 여부 안내",
            10: "배달 가능 지역 안내",
            11: "배달비 및 최소 주문 금액 안내",
            12: "예약 가능 여부 안내(전화/온라인)",
            13: "예약 취소/변경 절차 안내",
            14: "고객 대기 및 혼잡도 안내",
            15: "상점 주소 안내 및 지도 링크 전달",
            16: "테이블 배치 및 좌석 수 안내",
            17: "야외 테라스 또는 개별 룸 여부 안내",
            18: "멤버십 가입 / 혜택 안내",
            19: "포인트 적립 / 사용 관련 안내",
            20: "쿠폰 발행 / 사용 안내",
            21: "현재 진행 중인 이벤트 및 할인 혜택, 기간 정보 안내",
            22: "상점 관리자 연결 안내",
            23: "fallbackintent"
        }
        self.response_templates = {
            0: [
                "저희는 {STD_CATEGORY_NM} 등의 메뉴를 팔아요. 더 자세히 살펴보시라고 메뉴 링크를 문자로 발송해드렸어요."
            ],
            1: [
                "{STD_MENU_NM}의 가격은 {PRICE}입니다. 더 자세한 내용은 문자로 발송해드린 메뉴판을 확인해주세요."
            ],
            2: [
                "{INDI_TYPE_NM1} 메뉴를 {RECOMMEND_MENU} 메뉴로서 추천드려요."
            ],
            3: [
                "{LIMMIT_MENU} 메뉴가 {INDI_TYPE_NM3} 메뉴에요."
            ],
            4: [
                "저희 매장에서는 {ORDER_TYPE} 방식으로 주문이 가능해요."
            ],
            5: [
                "{PAYMNT_MN_CD} 방식의 결제가 가능하며 간편결제로는 {EASY_PAYMNT_TYPE_CD} 방식이 가능해요."
            ],
            6: [   # PAYMENT_AVAIL 처리하기
                "저희 매장에서는 {PAYMENT_NM_SPE}의 결제 방식이 {PAYMENT_AVAIL}"
            ],
            7: [
                "저희 {STORE_NM}의 영업시간은 {SALE_BGN_TM}부터 {SALE_END_TM}까지에요."
            ],
            8: [
                 "저희 매장은 {DAY_NM}에 영업{DAY_INFO}해요. {SALE_BGN_TM}부터 {SALE_END_TM} 사이에 방문해주세요."    # !!!!!!!!!!!! 영업시간을 info에 추가하기. 요일별로 영업시간이 다를 수 있음.
            ],
            9: [
                "저희 {STORE_NM}의 휴일은 {HOLIDAY_TYPE_CD}이에요."
            ],
            10: [
                "배달은 반경 {DLVR_ZONE_RADS} 이내의 지역만 가능해요. 자세한 사항은 매장에 문의해주세요."
            ]
            ,
            11: [
                "저희 매장의 배달 팁은 최소 {DLVR_TIP_AMT} 이며 {ORDER_BGN_AMT} 이상 주문해주셔야 배달이 가능해요."
            ],
            12: [
                "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요."
            ],
            13: [
                "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요."
            ],
            14: [
                "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요."
            ],
            15: [
                "저희 매장은 {ROAD_NM_ADDR}에 위치해 있습니다."
            ],
            16: [
                "총 테이블 수는 {TABLE_NM}개로 {MAX_PEOPLE}명 만큼 수용가능해요."
            ],
            17: [
                "해당 문의는 확인 후 전화하신 번호로 문자 안내 드릴게요."
            ],
            18: [
                "포인트 서비스나 쿠폰을 발급 받기 위해서는 회원 가입이 필요합니다. 회원가입은 스마트 주문 앱 또는 키오스크에서 가입 가능해요.",
            ],
            19: [
                "포인트 {USE_POSBL_BASE_POINT}점 부터 사용 가능해요."
            ],
            20: [
                "스탬프 {STAMP_TMS}개 적립 시 {COUPON_PACK_NM} 쿠폰을 발급해 주고 있어요."
            ],
            21: [
                "{EVENT_NM} 이벤트가 진행되고 있으며 {EVENT_BNEF_CND_CD} 동안 진행돼요."
            ],
            22: [
                "해당 문의는 확인 후 전화하신 번호로 안내 다시 전화 안내 드릴게요."
            ],
            23: [  # 일상대화??
                "해당 질문은 제가 답변드릴수 없어요."
            ]
        }



args = Args()

# === 학습된 의도 분류 모델 및 토크나이저 로드 === #
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
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
            result = df.query("STD_CATEGORY_NM != '없음' and STD_CATEGORY_NM != ''")[['STORE_NM', 'STD_CATEGORY_NM']].drop_duplicates()
        elif intent_id == 2:  # 인기 / 추천 메뉴 안내
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM1!= '없음'")[['STORE_NM', 'INDI_TYPE_NM1', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'RECOMMEND_MENU'}).drop_duplicates()
        elif intent_id == 3:    
            result = df.sample(frac=1, random_state=None).query("INDI_TYPE_NM3!= '없음'")[['STORE_NM', 'INDI_TYPE_NM3', 'STD_PROD_NM']].rename(columns={'STD_PROD_NM': 'LIMMIT_MENU'}).drop_duplicates()
        elif intent_id == 4:  # 주문 / 전달 방식 안내
            result = df.query("ORDER_TYPE != '없음' and ORDER_TYPE != ''")[['STORE_NM','ORDER_TYPE']].drop_duplicates()
        elif intent_id == 5:  # 결제 방법 안내
            result = df.query("PAYMNT_MN_CD != '없음' and EASY_PAYMNT_TYPE_CD != '없음'")[['STORE_NM','PAYMNT_MN_CD', 'EASY_PAYMNT_TYPE_CD']].drop_duplicates()
        elif intent_id == 7:  # 영업 시작/종료 시간 안내
             result = df.query("SALE_BGN_TM != '없음' and SALE_END_TM != '없음'")[['STORE_NM','SALE_BGN_TM', 'SALE_END_TM']].drop_duplicates()
        elif intent_id == 9:  # 휴일 정보 안내
            result = df.query("HOLIDAY_TYPE_CD != '없음' and HOLIDAY_TYPE_CD != ''")[['STORE_NM','HOLIDAY_TYPE_CD']].drop_duplicates()
        elif intent_id == 10:  # 배달 가능 지역 안내
            result = df.query("DLVR_ZONE_RADS != '없음' and DLVR_ZONE_RADS != ''")[['STORE_NM','DLVR_ZONE_RADS']].drop_duplicates()
        elif intent_id == 11:  # 배달비 및 최소 주문 금액 안내
            result = df.query("DLVR_TIP_AMT != '없음' and ORDER_BGN_AMT != '없음'")[['STORE_NM','DLVR_TIP_AMT', 'ORDER_BGN_AMT']].drop_duplicates()
        elif intent_id == 15:  # 상점 주소 안내
            result = df.query("ROAD_NM_ADDR != '없음' and ROAD_NM_ADDR != ''")[['STORE_NM', 'ROAD_NM_ADDR']].drop_duplicates()
        elif intent_id == 16:
            result = df.query("TABLE_NM != '없음' and TABLE_NM != ''")[['TABLE_NM']].drop_duplicates().assign(MAX_PEOPLE=lambda x: x['TABLE_NM'].astype(int) * 4)[['TABLE_NM', 'MAX_PEOPLE']]
        elif intent_id == 19:  # 포인트 사용 기준 안내
            result = df.query("USE_POSBL_BASE_POINT != '없음'")[['STORE_NM','USE_POSBL_BASE_POINT']].drop_duplicates()
        elif intent_id == 20:  # 스탬프 및 쿠폰 안내
            result = df.query("STAMP_TMS != '없음' and COUPON_PACK_NM != '없음'")[['STORE_NM','STAMP_TMS', 'COUPON_PACK_NM']].drop_duplicates()
        elif intent_id == 21:  # 이벤트 및 할인 혜택 안내
            result = df.sample(frac=1, random_state=None).query("EVENT_NM != '없음' and EVENT_BNEF_CND_CD != '없음'")[['STORE_NM','EVENT_NM', 'EVENT_BNEF_CND_CD']].drop_duplicates()
        elif intent_id in [12, 13, 14, 16, 17, 18, 22, 23]:  # 고정 응답이 필요한 intent_id
            print(f"Static response required for intent_id {intent_id}")
            return None
        elif intent_id == [1, 6, 8]:
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

# 유사도 계산 함수
def get_best_match(user_input, info_dict, yn=False):
    keys = list(info_dict.keys())
    
    values = list(info_dict.values())
    
    vectorizer = TfidfVectorizer().fit(keys + [user_input])
    key_vectors = vectorizer.transform(keys)
    user_vector = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_vector, key_vectors)[0]
   

    best_index = similarities.argmax()
    best_match = keys[best_index]
    print(best_match)
    best_value = values[best_index]
    best_score = similarities[best_index]
    
    
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
    '네이버페이': ''
}

holiday_info = {
    '월요일': {'영업여부': '안', '영업시작시간': '화요일에서 일요일 11시', '영업종료시간': '20시 30분'},  # !!!!!!!!!!! 안하는 경우에는 영업 시작 시간 부분에 영업하는 요일 붙이기!!
    '화요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '수요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '목요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '금요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '토요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'},
    '일요일': {'영업여부': '', '영업시작시간': '11시', '영업종료시간': '20시 30분'}
}

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

        if intent_id in [1, 6, 8]:
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

            elif intent_id == 6:
                payment_entities = [ent_text for ent_label, ent_text in ner_entities if ent_label == "PAYMENT"]
                if payment_entities:
                    target_entity = data_normalization(" ".join(payment_entities))

            elif intent_id == 8:
                day_entities = [ent_text for ent_label, ent_text in ner_entities if ent_label == "DAY"]
                if day_entities:
                    target_entity = data_normalization(" ".join(day_entities))
            
                    
        if intent_id in [1, 6, 8]:
            if intent_id == 1:
                select_info = menu_info
                
            elif intent_id == 6:
                select_info = payment_info
                
            elif intent_id == 8:
                select_info = holiday_info
                
            # info 딕셔너리에서 사용자 입력(user_message)와 가장 유사한 키를 찾음
            if intent_id == 6:  # 가능 여부 판단을 위해서 임계값 이상인 게 있는 경우는 있다고 판단, 없는 경우 없다고 판단.
                extracted_info = get_best_match(target_entity, select_info, yn=True)
                
                if extracted_info:
                    best_match, best_value, best_score, available = extracted_info  # 언패킹
                    
                    print(best_score)
            else:
                extracted_info = get_best_match(user_message, select_info)
                if extracted_info:
                    best_match, best_value, best_score = extracted_info # 언패킹
                    
                 
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
                print(final_data)

            elif intent_id == 6:
                # 특정 결제 수단 안내: {PAYMENT_NM_SPE}, {PAYMENT_AVAIL}
                final_data = {"PAYMENT_NM_SPE": best_match, "PAYMENT_AVAIL": available}
                final_data.update(result_data)  # SQL 결과 합치기
                response_text = generate_response(intent_id, final_data)

            elif intent_id == 8:
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
    app.run(host='0.0.0.0', port=1117, debug=False)
