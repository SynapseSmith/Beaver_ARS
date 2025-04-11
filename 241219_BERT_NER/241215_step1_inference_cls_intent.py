import os
os.environ["HF_HOME"] = "/home/user09/beaver/beaver_shared/data/cache"
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# GPU 설정 (사용할 GPU 번호 설정)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 설정값을 관리하는 클래스 정의
class Args:
    def __init__(self):
        # 저장된 모델 및 파일 경로 설정
        self.model_name = "klue/roberta-large"   # "klue/bert-base" "klue/roberta-large"
        self.output_dir = "/home/user09/beaver/data/shared_files/241215_BERT/checkpoint/klue_roberta_large_v2"  # klue_bert_base   klue_robert_large

# Args 객체 생성
args = Args()

# 라벨 매핑 딕셔너리
intents_dict = {
            0: "메뉴 카테고리 안내",
            1: "특정 상품 및 가격 안내", # 슬롯
            2: "상품에 대한 상세 및 추가 안내",
            3: "(뱃지) 인기메뉴",
            4: "(뱃지) 추천메뉴",
            5: "(뱃지) 대표메뉴",
            6: "(뱃지) 할인, 이벤트,",
            33: "(뱃지) 1+1 메뉴 문의", #---> 이거 33으로 수정함!! 
            7: "(뱃지) 신상 메뉴",
            8: "(뱃지) 한정 메뉴",
            9: "(뱃지) 매운맛",
            10: "주문 방식 안내 (키오스크, 테이블오더, 스마트주문 등)",
            11: "주문한 상품 전달 방식 안내 (내점, 포장, 배달 등)",  #!!!
            12: "결제 방법 안내(현금, 카드, 간편 결제 등)", #!!!
            13: "특정 결제 방법 상세 안내", #!!!!!   # 슬롯
            14: "결제 방법 추가 안내",   #이건 더 자세히 알려줘
            15: "영업 시간 안내",  # !!! 범용적인 영업
            16: "영업 시간 상세 안내",  # 자세하게 알려줘
            17: "브레이크 타임 안내",  # 만들기 
            18: "브레이크 타임 상세 안내",  #만들기기
            19: "휴무일 안내 (정기 - 임시휴무/주말/공휴일)",  
            20: "휴무일 상세 안내 (정기 - 임시휴무/주말/공휴일)",  # 주말, 평일 운영 여부
            21: "영업시간 및 휴무일 추가 안내",  # 자세히 알려줘
            22: "특정 요일에 대한 영업 여부",   #!!! # 슬롯
            23: "배달 가능 지역 안내",   # !!!
            24: "배달비 및 최소 주문 금액 안내", #!!!
            25: "배달 추가 안내", # 자세히
            26: "테이블 점유 안내",   # 만들기
            27: "테이블 점유 추가 안내",  # 더 자세히
            28: "상점 주소 안내 및 지도 링크 전달",  # !!!
            29: "대중교통 이용 방법 안내",  # !!!-> fallback --> 키워드 기반으로 생성. 버스, 지하철, 대중교통, 택시
            30: "근처 랜드마크 안내",  # !!!
            31: "테이블 및 좌석 수 안내",   # !!!
            32: "야외 테라스 또는 개별 룸 여부 안내",  # !!!
            # : "상점 규모 및 시설 추가 안내",  # 룸, 야외 더 자세히 알려줘
            34: "멤버십 가입 안내",  # 가입 되나요
            35: "멤버십 혜택 안내",  # 가입 시 혜택
            36: "포인트 적립 안내",  # !!!
            37: "포인트 사용 안내",  # !!!
            38: "쿠폰 발행 안내",  # !!!
            39: "쿠폰 사용 안내",  # !!!
            40: "멤버십 및 쿠폰에 대한 추가 안내",  # 더 자세히
            41: "현재 진행 중인 이벤트 안내",   # (이벤트 종류)진행중인 이벤트 있나요
            42: "현재 진행 중인 이벤트 상세 안내",   # (이벤트 설명 요청) 기간, 어떤 내용 
            43: "현재 진행 중인 이벤트 추가 안내",   # 더 자세히
            44: "CallBackIntent", # !!!!!
            45: "감사 인텐트",
            46: "인사 인텐트",
            47: "FallBackIntent"
        }

# 저장된 모델 및 토크나이저 불러오기
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
model.to(device)

# 예측 예시 입력
test_texts = ["신메뉴 알려줘.", "월요일 영업시간 몇시부터 몇시까지인지 알려줘", "예약을 취소하려면 어떻게 하나요?", '봉고차인데 가능할까요']

# 입력 데이터 토크나이즈 및 텐서로 변환
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
test_encodings = {k: v.to(device) for k, v in test_encodings.items()}

# 모델 추론
print("Performing inference...")
with torch.no_grad():
    outputs = model(**test_encodings)
preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

# 예측값 변환 및 출력
predicted_intents = [intents_dict[label] for label in preds]
for text, intent in zip(test_texts, predicted_intents):
    print(f"Input: {text} -> Predicted Intent: {intent}")
