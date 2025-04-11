import os
import pandas as pd
import re
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
    DataCollatorForTokenClassification  # 이 줄을 추가
)

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@dataclass
class Args:
    # 데이터 파일 경로
    conll_file_path: str = '/home/user09/beaver/data/shared_files/241218_NER/data/step1_labeled_data.conll'
    
    # 모델 및 토크나이저 설정
    model_name: str = "klue/roberta-large"
    
    # 학습 관련 설정
    output_model_dir: str = "/home/user09/beaver/data/shared_files/241218_NER/ner_checkpoint"
    logging_dir: str = './logs'
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 5
    weight_decay: float = 0.01
    save_total_limit: int = 2
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    run_name: str = "ner_training_run"
    
    # 랜덤 시드
    seed: int = 42

# 설정 인스턴스 생성
args = Args()

# ------------------------------
# 1. 랜덤 시드 설정 (재현성 확보)
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ------------------------------
# 2. 데이터 로드 및 전처리
# ------------------------------
def read_conll(file_path: str):
    """
    CoNLL 형식의 파일을 읽어들여 문장과 라벨을 분리합니다.

    Parameters:
    - file_path (str): CoNLL 파일의 경로

    Returns:
    - sentences (List[List[str]]): 문장의 토큰 리스트
    - labels (List[List[str]]): 각 토큰에 대한 라벨 리스트
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    sentences = []
    labels = []
    tokens = []
    ents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(ents)
                    tokens = []
                    ents = []
                continue
            splits = line.split('\t')
            if len(splits) != 2:
                tokens.append(splits[0])
                ents.append('O')
            else:
                token, label = splits
                tokens.append(token)
                ents.append(label)
        # 마지막 문장 추가
        if tokens:
            sentences.append(tokens)
            labels.append(ents)
    return sentences, labels

# 데이터 읽기
try:
    sentences, labels = read_conll(args.conll_file_path)
except FileNotFoundError as e:
    print(e)
    exit(1)

# DataFrame으로 변환
df = pd.DataFrame({'tokens': sentences, 'labels': labels})

# 데이터 검증: 일부 데이터 출력
print("데이터 검증:")
for i in range(min(3, len(df))):
    print(f"Utterance {i+1}: {' '.join(df['tokens'][i])}")
    print(f"Labels {i+1}: {df['labels'][i]}\n")

# ------------------------------
# 3. 라벨 인코딩
# ------------------------------
# 라벨 리스트 생성
unique_labels = sorted(set(label for doc_labels in labels for label in doc_labels))
label_list = unique_labels

# 다중 토큰 엔티티 처리를 위한 I-ENTITY 라벨 추가 (필요 시)
# 만약 데이터에 I-ENTITY 라벨이 없다면, 아래 코드를 통해 추가할 수 있습니다.
# 예: ['B-MENU', 'I-MENU', 'B-PAYMENT', 'I-PAYMENT', 'B-DAY', 'I-DAY', 'O']
# 필요하지 않다면 주석 처리

# entity_types = set(label.split('-')[1] for label in label_list if label != 'O')
# for entity in entity_types:
#     if f'I-{entity}' not in label_list:
#         label_list.append(f'I-{entity}')

label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

def encode_labels(example):
    """
    각 예제의 라벨을 정수로 인코딩합니다.

    Parameters:
    - example (dict): Dataset의 한 예제

    Returns:
    - example (dict): 인코딩된 라벨을 포함한 예제
    """
    example['labels'] = [label_to_id[label] for label in example['labels']]
    return example

dataset = Dataset.from_pandas(df)
dataset = dataset.map(encode_labels)

# 라벨 리스트 확인
print("라벨 리스트:")
print(label_list)
print("\n")

# ------------------------------
# 4. 데이터셋 분할
# ------------------------------
# 데이터셋 분할 (80% 학습, 20% 검증)
dataset = dataset.train_test_split(test_size=0.2)

# ------------------------------
# 5. 모델 및 토크나이저 로드
# ------------------------------
model_name = args.model_name

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델 로드 (레이블 수에 맞게 num_labels 설정)
num_labels = len(label_list)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# ------------------------------
# 6. 데이터 토큰화 및 라벨 정렬
# ------------------------------
def tokenize_and_align_labels(examples):
    """
    토큰화된 입력과 라벨을 정렬합니다.

    Parameters:
    - examples (dict): Dataset의 배치 예제

    Returns:
    - tokenized_inputs (dict): 토큰화된 입력과 정렬된 라벨
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,  # 길이 제한
        is_split_into_words=True,  # 이미 토큰화된 입력
        padding='max_length',  # 고정 길이로 패딩 추가
        max_length=128  # 필요한 경우 적절히 설정
    )

    labels = examples['labels']
    aligned_labels = []

    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # 서브워드 토큰 무시
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    tokenized_inputs['labels'] = aligned_labels
    return tokenized_inputs


def validate_data(dataset):
    for i, example in enumerate(dataset):
        if len(example['tokens']) != len(example['labels']):
            print(f"길이 불일치: {i}-번째 데이터")
            print(f"Tokens: {example['tokens']}")
            print(f"Labels: {example['labels']}")
            raise ValueError("토큰과 라벨 길이가 일치하지 않습니다.")



# 데이터셋 토큰화 및 정렬
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# 불필요한 컬럼 제거
tokenized_dataset = tokenized_dataset.remove_columns(["tokens"])
tokenized_dataset.set_format("torch")


# 데이터 샘플 출력
print("토큰화된 데이터 샘플 확인:")
print(tokenized_dataset['train'][0])


print("모델 입력 데이터 샘플:")
print(tokenized_dataset['train'][0])


# ------------------------------
# 7. 평가 메트릭 정의
# ------------------------------
def compute_metrics(p):
    """
    모델의 예측 결과를 기반으로 평가 메트릭을 계산합니다.

    Parameters:
    - p (tuple): (predictions, labels)

    Returns:
    - metrics (dict): F1, Precision, Recall 및 분류 보고서
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, label in zip(predictions, labels):
        temp_pred = []
        temp_label = []
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
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "report": report
    }

# ------------------------------
# 8. 훈련 인수 설정
# ------------------------------
training_args = TrainingArguments(
    output_dir=args.output_model_dir,
    evaluation_strategy=args.evaluation_strategy,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    save_total_limit=args.save_total_limit,
    logging_dir=args.logging_dir,
    logging_steps=args.logging_steps,
    save_strategy=args.save_strategy,
    load_best_model_at_end=args.load_best_model_at_end,
    metric_for_best_model=args.metric_for_best_model,
    # run_name=args.run_name,
    report_to=["none"]
)

# ------------------------------
# 9. Trainer 초기화 및 모델 학습
# ------------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()

# ------------------------------
# 10. 모델 평가 및 저장
# ------------------------------
# 모델 평가
results = trainer.evaluate()

# 평가 결과 출력
print("Evaluation Results:")
print(f"F1 Score: {results['eval_f1']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall: {results['eval_recall']:.4f}")
print("Classification Report:")
print(results['eval_report'])

# 모델 및 토크나이저 저장
output_model_dir = args.output_model_dir
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"모델이 '{output_model_dir}' 디렉토리에 저장되었습니다.")

# ------------------------------
# 11. 모델 테스트 (선택 사항)
# ------------------------------
# NER 파이프라인 로드
ner_pipeline = pipeline(
    "ner",
    model=os.path.join(output_model_dir),
    tokenizer=os.path.join(output_model_dir),
    aggregation_strategy="simple"  # 엔티티 그룹화를 위한 설정
)

# 테스트 발화
test_utterance = "토요일에 피자 가격 얼마야? 네이버 페이로 결제할 수 있나요?"

# NER 수행
entities = ner_pipeline(test_utterance)

# 결과 출력
print("NER 결과:")
for entity in entities:
    print(f"엔티티: {entity['word']}, 라벨: {entity['entity_group']}, 시작: {entity['start']}, 종료: {entity['end']}")

# 추가적인 테스트 발화
test_utterances = [
    "금요일에 현금으로 결제할 수 있나요?",
    "스파게티 가격이 얼마인지 알려주세요.",
    "수요일에 예약하려면 어떻게 해야 하나요?"
]

for utterance in test_utterances:
    entities = ner_pipeline(utterance)
    print(f"\nUtterance: {utterance}")
    print("NER 결과:")
    for entity in entities:
        print(f"엔티티: {entity['word']}, 라벨: {entity['entity_group']}, 시작: {entity['start']}, 종료: {entity['end']}")
