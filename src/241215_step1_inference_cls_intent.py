import os
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Argument parser 설정
parser = argparse.ArgumentParser(description="Intent Classification Inference")
parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
parser.add_argument("--text", type=str, required=True, help="Text to classify")
args = parser.parse_args()

# 라벨 매핑 딕셔너리
intents_dict = {
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

# 저장된 모델 및 토크나이저 불러오기
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path, local_files_only=True)
model.to(device)

# 예측 입력
test_texts = [args.text]

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
