# 모델 학습 가이드

## 목차
1. [환경 준비](#환경-준비)
2. [데이터셋 준비](#데이터셋-준비)
3. [Intent Classification 학습](#intent-classification-학습)
4. [NER 모델 학습](#ner-모델-학습)
5. [모델 평가](#모델-평가)
6. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
7. [모델 배포](#모델-배포)

---

## 환경 준비

### 1. 하드웨어 요구사항

#### 최소 사양
- **CPU**: Intel i5 이상 (또는 동급)
- **RAM**: 16GB 이상
- **Storage**: 50GB 여유 공간
- **학습 시간**: ~4시간 (CPU)

#### 권장 사양
- **CPU**: Intel i7 이상 (또는 동급)
- **RAM**: 32GB 이상
- **GPU**: NVIDIA RTX 3060 이상 (VRAM 12GB+)
- **Storage**: 100GB 여유 공간 (SSD 권장)
- **학습 시간**: ~30분 (GPU)

### 2. 소프트웨어 환경

```bash
# Python 버전 확인
python --version  # 3.8 이상 필요

# CUDA 설치 확인 (GPU 사용 시)
nvidia-smi
nvcc --version  # CUDA 11.8 이상 권장

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# GPU용 PyTorch 설치 (필요시)
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Weights & Biases 설정 (선택)

```bash
# wandb 로그인
wandb login

# API 키 입력 (https://wandb.ai/authorize 에서 확인)
```

---

## 데이터셋 준비

### Intent Classification 데이터

#### 1. 데이터 형식

**CSV 형식** (`data/user_intent_v4.csv`)
```csv
text,intent,intent_num
"메뉴판 좀 보여주세요",메뉴 카테고리 안내,0
"떡볶이 얼마예요?",특정 상품 및 가격 안내,1
"카드 결제 되나요?",결제 방법 안내,13
"영업시간이 언제예요?",영업 시간 안내,16
```

#### 2. 데이터 수집 가이드

1. **다양성 확보**
   - 동일 의도에 대해 다양한 표현 수집
   - 구어체, 문어체, 줄임말 등 포함
   - 오타가 포함된 데이터도 일부 포함

2. **균형 유지**
   - 각 클래스당 최소 50개 샘플 권장
   - 불균형이 심한 경우 Data Augmentation 고려

3. **품질 관리**
   - 중복 데이터 제거
   - 라벨링 오류 검수
   - 애매한 케이스는 여러 annotator가 검토

#### 3. 데이터 증강 (Augmentation)

```python
# Back-translation을 이용한 증강
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, src_lang='ko', pivot_lang='en'):
    """한국어 → 영어 → 한국어 역번역"""
    # Ko → En
    model_ko_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    tokenizer_ko_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    
    # En → Ko
    model_en_ko = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ko')
    tokenizer_en_ko = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ko')
    
    # 번역 수행
    tokens = tokenizer_ko_en(text, return_tensors='pt')
    translated_en = model_ko_en.generate(**tokens)
    en_text = tokenizer_ko_en.decode(translated_en[0], skip_special_tokens=True)
    
    tokens = tokenizer_en_ko(en_text, return_tensors='pt')
    translated_ko = model_en_ko.generate(**tokens)
    ko_text = tokenizer_en_ko.decode(translated_ko[0], skip_special_tokens=True)
    
    return ko_text

# 사용 예
original = "떡볶이 가격이 얼마예요?"
augmented = back_translate(original)
print(augmented)  # "떡볶이는 얼마예요?" (변형된 표현)
```

### NER 데이터

#### 1. CoNLL 형식

**파일 구조** (`data/NER_labeled_data_v2.conll`)
```conll
떡볶이	B-MENU
가격이	O
얼마예요	O

김밥	B-MENU
이랑	O
라면	B-MENU
주세요	O

카드	B-PAYMENT
결제	I-PAYMENT
되나요	O
```

#### 2. 라벨링 가이드

| 엔티티 타입 | 설명 | 예시 |
|------------|------|------|
| `B-MENU` | 메뉴명 시작 | 떡볶이, 김밥, 치즈떡볶이 |
| `I-MENU` | 메뉴명 계속 | **치즈**떡볶이 (치즈=B, 떡볶이=I) |
| `B-PAYMENT` | 결제수단 시작 | 카드, 현금, 페이 |
| `I-PAYMENT` | 결제수단 계속 | **신용**카드 (신용=B, 카드=I) |
| `B-DAY` | 요일 | 월요일, 주말, 평일 |
| `O` | 기타 | 나머지 모든 토큰 |

#### 3. 라벨링 도구

```python
# 간단한 라벨링 인터페이스
def label_sentence(sentence):
    """대화형 라벨링"""
    tokens = sentence.split()
    labels = []
    
    print(f"\n문장: {sentence}")
    print("0: O, 1: B-MENU, 2: I-MENU, 3: B-PAYMENT, 4: I-PAYMENT, 5: B-DAY")
    
    for token in tokens:
        label = input(f"{token}: ")
        label_map = {
            '0': 'O', '1': 'B-MENU', '2': 'I-MENU',
            '3': 'B-PAYMENT', '4': 'I-PAYMENT', '5': 'B-DAY'
        }
        labels.append(label_map.get(label, 'O'))
    
    # CoNLL 형식으로 저장
    with open('labeled_data.conll', 'a', encoding='utf-8') as f:
        for token, label in zip(tokens, labels):
            f.write(f"{token}\t{label}\n")
        f.write("\n")  # 문장 구분
```

---

## Intent Classification 학습

### 1. 학습 스크립트 실행

```bash
cd 241215_BERT
python 241215_step1_train_cls_intent.py
```

### 2. 주요 하이퍼파라미터

```python
class Args:
    # 모델
    model_name = "klue/roberta-large"  # 또는 "klue/bert-base"
    num_labels = 48
    
    # 학습
    num_train_epochs = 20
    per_device_train_batch_size = 16  # GPU 메모리에 따라 조정
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_steps = 100
    
    # 평가
    evaluation_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True
    metric_for_best_model = "f1"
    
    # 데이터
    test_size = 0.2
    max_length = 128
```

### 3. 학습 모니터링

#### Weights & Biases 대시보드
- **Loss Curve**: 훈련/검증 손실 추이
- **Accuracy**: 에폭별 정확도
- **F1-Score**: 클래스별 F1 점수
- **Learning Rate**: 학습률 스케줄

#### 콘솔 출력
```
Epoch 1/20
Train Loss: 2.453, Train Acc: 0.412, Val Loss: 1.823, Val Acc: 0.583
Epoch 2/20
Train Loss: 1.612, Train Acc: 0.651, Val Loss: 1.234, Val Acc: 0.712
...
Epoch 20/20
Train Loss: 0.187, Train Acc: 0.981, Val Loss: 0.203, Val Acc: 0.957
```

### 4. Early Stopping

조기 종료를 위한 설정:

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

---

## NER 모델 학습

### 1. 학습 스크립트 실행

```bash
cd 241218_NER
python 241218_step1_ner_train_i_tagging.py
```

### 2. 학습 설정

```python
@dataclass
class Args:
    # 데이터
    conll_file_path: str = 'data/NER_labeled_data_v2.conll'
    test_size: float = 0.2
    
    # 모델
    model_name: str = "klue/roberta-large"
    
    # 학습
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    num_train_epochs: int = 2  # NER은 보통 2-5 에폭
    
    # 출력
    output_model_dir: str = "ner_checkpoint"
```

### 3. seqeval 평가

NER은 entity-level 평가를 사용합니다:

```python
from seqeval.metrics import classification_report

# 예측 결과
y_true = [['O', 'B-MENU', 'O', 'O']]
y_pred = [['O', 'B-MENU', 'O', 'O']]

# 평가
report = classification_report(y_true, y_pred)
print(report)
```

출력:
```
              precision    recall  f1-score   support

        MENU       1.00      1.00      1.00         1

   micro avg       1.00      1.00      1.00         1
   macro avg       1.00      1.00      1.00         1
```

---

## 모델 평가

### 1. Intent Classification 평가

```bash
python 241215_step1_evaluation_cls_intent.py
```

#### Confusion Matrix 분석

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Intent Classification Confusion Matrix')
plt.savefig('confusion_matrix.png')
```

#### Per-Class Metrics

```python
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=intent_names)
print(report)
```

### 2. NER 평가

```bash
python 241218_step2_ner_evaluation.py
```

#### Entity-Level Metrics

```python
from seqeval.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

---

## 하이퍼파라미터 튜닝

### 1. Grid Search

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [8, 16, 32],
    'epochs': [10, 15, 20]
}

best_f1 = 0
best_params = None

for params in ParameterGrid(param_grid):
    # 모델 학습
    model = train_model(**params)
    
    # 평가
    f1 = evaluate_model(model)
    
    if f1 > best_f1:
        best_f1 = f1
        best_params = params

print(f"Best F1: {best_f1:.4f}")
print(f"Best Params: {best_params}")
```

### 2. 실험 결과 예시

| Learning Rate | Batch Size | Epochs | F1-Score |
|--------------|-----------|--------|----------|
| 1e-5 | 16 | 20 | 0.943 |
| 2e-5 | 16 | 20 | **0.957** |
| 3e-5 | 16 | 20 | 0.951 |
| 2e-5 | 8 | 20 | 0.948 |
| 2e-5 | 32 | 20 | 0.952 |

---

## 모델 배포

### 1. 모델 저장

```python
# 모델 저장
model.save_pretrained('./final_model')
tokenizer.save_pretrained('./final_model')

# 설정 저장
import json

config = {
    'model_name': 'klue/roberta-large',
    'num_labels': 48,
    'max_length': 128,
    'intent_dict': intent_dict
}

with open('./final_model/config.json', 'w') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
```

### 2. 모델 로드

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 로드
model = AutoModelForSequenceClassification.from_pretrained('./final_model')
tokenizer = AutoTokenizer.from_pretrained('./final_model')

# 추론
text = "떡볶이 가격이 얼마예요?"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=-1).item()
```

### 3. ONNX 변환 (최적화)

```python
import torch.onnx

# 더미 입력
dummy_input = tokenizer("테스트", return_tensors='pt')

# ONNX 변환
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    'model.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch'}
    }
)
```

---

## 문제 해결

### Out of Memory (OOM)

```python
# 배치 크기 줄이기
per_device_train_batch_size = 8  # 16 → 8

# Gradient Accumulation
gradient_accumulation_steps = 2  # 실질적으로 batch_size * 2

# Mixed Precision Training
fp16 = True
```

### 과적합 (Overfitting)

```python
# Dropout 증가
dropout = 0.3  # 기본값: 0.1

# Weight Decay 증가
weight_decay = 0.05  # 기본값: 0.01

# Early Stopping
early_stopping_patience = 3
```

### 학습이 느린 경우

```python
# 더 작은 모델 사용
model_name = "klue/bert-base"  # roberta-large 대신

# DataLoader Workers 증가
dataloader_num_workers = 4

# 캐싱 활용
dataset = dataset.map(preprocess, batched=True, cache_file_name='cache.arrow')
```

---

**문서 버전**: 1.0  
**최종 수정일**: 2025-01-02  
**작성자**: Beaver ARS Team
