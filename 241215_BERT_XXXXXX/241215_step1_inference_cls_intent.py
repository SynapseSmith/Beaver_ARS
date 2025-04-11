import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# GPU 설정 (사용할 GPU 번호 설정)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 설정값을 관리하는 클래스 정의
class Args:
    def __init__(self):
        # 저장된 모델 및 파일 경로 설정
        self.model_name = "klue/roberta-large"   # "klue/bert-base" "klue/roberta-large"
        self.output_dir = "C:/Users/user/PycharmProjects/pythonProject/beaver/241215_BERT_XXXXXX/checkpoint/klue_roberta_large"  # klue_bert_base   klue_robert_large

# Args 객체 생성
args = Args()

# 라벨 매핑 딕셔너리
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

# 저장된 모델 및 토크나이저 불러오기
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
model.to(device)

# 예측 예시 입력
test_texts = ["메뉴 종류를 알려주세요.", "결제 방법을 알려주세요.", "예약을 취소하려면 어떻게 하나요?", '봉고차인데 가능할까요']

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
