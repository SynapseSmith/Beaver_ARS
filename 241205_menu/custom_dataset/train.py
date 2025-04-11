import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# GPU 설정 (사용할 GPU 번호 설정)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 디바이스 설정 (GPU 사용 가능 여부 판단)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 로드 (의도 분류에 맞는 데이터셋 사용)
raw_datasets = load_dataset("klue", "ynat")  # 필요에 따라 다른 데이터셋으로 변경 가능

model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = 7   # 의도(intent) 개수

def preprocess_function(examples):  # 전처리 함수
    return tokenizer(examples["title"], padding=False, truncation=True, max_length=128)

# 데이터셋 전처리
encoded_datasets = raw_datasets.map(preprocess_function, batched=True)

# 라벨 열을 'labels'로 변경
encoded_datasets = encoded_datasets.rename_column("label", "labels")

# 불필요한 열 제거 및 포맷 설정
encoded_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 데이터 콜레이터 초기화 (동적 패딩을 위해)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 모델 초기화 및 디바이스로 이동
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

# 평가 지표 함수 정의
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="C:/Users/user/PycharmProjects/pythonProject/beaver/241205_menu/custom_dataset/results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # 필요에 따라 에포크 수 조정
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator  # 데이터 콜레이터 추가
)

# 학습
trainer.train()

# 학습한 모델 저장
output_dir = '/home/user09/beaver/data/shared_files/241205_menu/custom_dataset/bert-topic-cls'  # 허깅페이스 모델 저장 경로
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# 평가
eval_results = trainer.evaluate()
print("Eval results:", eval_results)

# 예측
test_texts = ["정부, 가계부채 대책 강화", "프로야구 KIA, 플레이오프 진출 확정"]
test_encodings = tokenizer(test_texts, truncation=True, padding=False, max_length=128, return_tensors="pt")

# 데이터 콜레이터를 사용하여 동적 패딩 적용
test_encodings = data_collator(test_encodings)

# 입력 텐서 디바이스 이동
test_encodings = {k: v.to(device) for k, v in test_encodings.items()}

# 모델 추론
with torch.no_grad():
    outputs = model(**test_encodings)
preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()  # CPU로 이동 후 numpy 변환
print("예측 라벨:", preds)
