import os
import torch
import json
import argparse
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


# Device 설정 - RTX 5090 GPU 사용
import warnings
warnings.filterwarnings('ignore', message='.*sm_120.*')  # sm_120 경고 무시

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Argument parser 설정
parser = argparse.ArgumentParser(description="Train Intent Classifier")
parser.add_argument("--data_path", type=str, required=True, help="Path to training data CSV")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model")
parser.add_argument("--model_name", type=str, default="klue/roberta-large", help="Pretrained model name")
parser.add_argument("--num_labels", type=int, default=48, help="Number of intent labels")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")
args = parser.parse_args()

# 데이터셋 로드
df = pd.read_csv(args.data_path)

# 데이터셋 변환: Hugging Face Dataset 포맷으로 변환
dataset = Dataset.from_pandas(df)

# 데이터셋 나누기 (Train / Validation / Test)
train_test_split = dataset.train_test_split(test_size=args.test_size, seed=42)
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
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir=args.logging_dir,
    logging_steps=50,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    num_train_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # GPU에서만 Mixed precision 사용
    optim="adamw_torch",
    gradient_checkpointing=True,  # Save memory
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