import os
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
from sklearn.metrics import accuracy_score, f1_score


# GPU 설정 (사용할 GPU 번호 설정)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 설정값을 관리하는 클래스 정의
class Args:
    def __init__(self):
        # 모델 및 파일 경로 설정
        self.model_name = "klue/roberta-large"   # "klue/bert-base" "klue/roberta-large"
        self.output_dir = "/home/user09/beaver/data/shared_files/241215_BERT/checkpoint/klue_roberta_large_v4"  # klue_bert_base    klue_roberta_large
        self.file_path = "/home/user09/beaver/data/shared_files/241215_BERT/data/user_intent_v4.csv"

        # 데이터 관련 설정
        self.num_labels = 24  # 0~23까지 총 24개의 라벨
        self.test_size = 0.2  # 테스트 데이터 비율
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

# 모델 및 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
model.to(device)

# 평가 지표 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

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
    greater_is_better=True,
    report_to=[]
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

# 학습
trainer.train()

# 모델 저장
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# 평가
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)