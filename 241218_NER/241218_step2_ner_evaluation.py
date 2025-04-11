import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"

# 설정 클래스
class Args:
    def __init__(self):
        self.model_dir = "C:/Users/user/PycharmProjects/pythonProject/beaver/241218_NER/ner_checkpoint2"
        self.data_path = "C:/Users/user/PycharmProjects/pythonProject/beaver/241218_NER/data/NER_labeled_data_v2.conll"
        self.output_xlsx_path = "C:/Users/user/PycharmProjects/pythonProject/beaver/241218_NER/data/NER_test_results.xlsx"
        self.random_seed = 42

args = Args()

# 랜덤 시드 설정
def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.random_seed)

# CoNLL 형식의 데이터 읽기 함수
def read_conll(file_path):
    sentences, labels = [], []
    tokens, ents = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(ents)
                    tokens, ents = [], []
                continue
            splits = line.split('\t')
            tokens.append(splits[0])
            ents.append(splits[1])
    if tokens:
        sentences.append(tokens)
        labels.append(ents)
    return sentences, labels

# 데이터 읽기
sentences, labels = read_conll(args.data_path)

# 데이터프레임 생성
df = pd.DataFrame({"tokens": sentences, "labels": labels})

# 라벨 리스트 정의
label_list = ["O", "B-MENU", "I-MENU", "B-PAYMENT", "I-PAYMENT", "B-DAY"]

# 라벨 매핑
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# 라벨 인코딩
def encode_labels(labels):
    return [[label_to_id[label] for label in seq] for seq in labels]

df["encoded_labels"] = encode_labels(df["labels"])

# 데이터셋 분할
train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.random_seed)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

# 데이터 전처리 함수
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )
    labels = examples["encoded_labels"]
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 모델 로드
model = AutoModelForTokenClassification.from_pretrained(args.model_dir, num_labels=len(label_list))

# 평가 메트릭 정의
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels, true_predictions = [], []
    for pred, label in zip(predictions, labels):
        temp_pred, temp_label = [], []
        for p, l in zip(pred, label):
            if l != -100:
                temp_pred.append(id_to_label[p])
                temp_label.append(id_to_label[l])
        true_labels.append(temp_label)
        true_predictions.append(temp_pred)
    f1 = f1_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions)
    return {"f1": f1, "precision": precision, "recall": recall, "report": report}

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir=args.model_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_eval_batch_size=16,
    save_total_limit=1,
    load_best_model_at_end=True,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 평가 수행
results = trainer.evaluate()

# 평가 결과 출력
print("Evaluation Results:")
print(f"F1 Score: {results['eval_f1']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall: {results['eval_recall']:.4f}")
print("Classification Report:")
print(results["eval_report"])

# 테스트 데이터 예측
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=2)

# 결과 데이터프레임 생성
test_results = []
for i, example in enumerate(test_dataset):
    tokens = example["tokens"]
    actual = [id_to_label[l] for l in example["labels"] if l != -100]
    predicted = [id_to_label[p] for p, l in zip(predicted_labels[i], example["labels"]) if l != -100]
    test_results.append({"tokens": tokens, "actual": actual, "predicted": predicted})

results_df = pd.DataFrame(test_results)

# 엑셀 저장
results_df.to_excel(args.output_xlsx_path, index=False)
print(f"Test results saved to {args.output_xlsx_path}")
