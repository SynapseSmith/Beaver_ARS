import os
os.environ["HF_HOME"] = "/home/user09/beaver/beaver_shared/data/cache"
import torch
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from evaluate import load
from sklearn.metrics import accuracy_score, f1_score, classification_report

# GPU 설정 (사용할 GPU 번호 설정)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 설정값을 관리하는 클래스 정의
class Args:
    def __init__(self):
        # 모델 및 파일 경로 설정
        self.model_name = "klue/roberta-large"   # "klue/bert-base" "klue/roberta-large"
        self.output_dir = "/home/user09/beaver/data/shared_files/241219_BERT_NER/checkpoint/klue_roberta_large_v8"
        self.file_path = "/home/user09/beaver/data/shared_files/241219_BERT_NER/data/user_intent_v13.csv"
        self.output_xlsx_path  = "/home/user09/beaver/data/shared_files/241219_BERT_NER/data/intent_test_results_v8.xlsx"

        # 데이터 관련 설정
        self.num_labels = 48  # 0~21까지 총 22개의 라벨
        self.test_size = 0.1  # 테스트 데이터 비율
        self.random_seed = 42  # 데이터셋 분할 시 사용할 랜덤 시드

        # 학습 관련 설정
        self.num_train_epochs = 20  # 에폭 수
        self.per_device_train_batch_size = 16  # 학습 배치 크기
        self.per_device_eval_batch_size = 16  # 평가 배치 크기
        self.weight_decay = 0.01  # 가중치 감쇠율
        self.logging_steps = 1000  # 로깅 간격
        self.evaluation_strategy = "epoch"  # 평가 전략
        self.save_strategy = "epoch"  # 체크포인트 저장 전략
        self.logging_dir = "./logs"  # 로그 저장 디렉토리

# Args 객체 생성
args = Args()

# Intent dictionary 정의
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

# 데이터셋 로드
df = pd.read_csv(args.file_path)

# 데이터셋 변환: Hugging Face Dataset 포맷으로 변환
dataset = Dataset.from_pandas(df)

# 데이터셋 나누기 (Train / Validation / Test)
train_test_split = dataset.train_test_split(test_size=args.test_size, seed=args.random_seed)
final_datasets = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

# 데이터 전처리 함수
def preprocess_function(examples):
    return tokenizer(examples["user"], padding=False, truncation=True, max_length=128)

# 데이터셋 전처리
encoded_datasets = final_datasets.map(preprocess_function, batched=True)
encoded_datasets = encoded_datasets.rename_column("intent_num", "labels")  # 라벨을 'labels'로 변경
encoded_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 데이터 콜레이터 설정
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, num_labels=args.num_labels)
model.to(device)
print("args.num_labels:", args.num_labels)
print("len(intents_dict):", len(intents_dict))

# 평가 지표 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print("Unique labels:", set(labels))
    print("Unique predictions:", set(predictions))
    print("Number of unique labels:", len(set(labels)))
    print("Number of unique predictions:", len(set(predictions)))
    # classification_report를 문자열 형식으로 생성
    report = classification_report(
        labels,
        predictions,
        target_names = [intents_dict[i] for i in range(args.num_labels)],
        labels=list(range(args.num_labels)),
        output_dict=False  # 문자열 형식으로 반환
    )
    # 콘솔에 보기 좋게 출력
    print("Classification Report:\n", report)
    
    # Accuracy와 Weighted F1-Score를 계산 (옵션)
    acc = accuracy_score(labels, predictions)
    weighted_f1 = f1_score(labels, predictions, average="weighted")
    
    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1
    }

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy=args.evaluation_strategy,
    save_strategy=args.save_strategy,
    load_best_model_at_end=True,
    logging_dir=args.logging_dir,
    logging_steps=args.logging_steps,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# 평가 실행
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# 테스트 데이터셋 준비
test_dataset = encoded_datasets["validation"]

# 예측 수행
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids

# 예측 값 계산
predicted_labels = np.argmax(logits, axis=-1)
is_correct = (predicted_labels == labels)  # True/False 비교

# 원본 데이터 가져오기 (토큰화 전 데이터 사용)
original_test_data = final_datasets["validation"]

# 테스트 데이터와 결과를 리스트로 저장
results = []
for i in range(len(test_dataset)):
    item = {
        "user": original_test_data[i]["user"],  # 입력 텍스트
        "true_label": intents_dict[labels[i]],  # 실제 라벨을 문자열로 변환
        "predicted_label": intents_dict[predicted_labels[i]],  # 예측 라벨을 문자열로 변환
        "correct": bool(is_correct[i])  # True/False
    }
    results.append(item)

# 결과를 Pandas DataFrame으로 변환
results_df = pd.DataFrame(results)

# 결과를 엑셀 파일로 저장
output_xlsx_path = args.output_xlsx_path
results_df.to_excel(output_xlsx_path, index=False)
print(f"Test results saved to {output_xlsx_path}")

# 데이터프레임 출력 (샘플 5개)
print(results_df.head())
