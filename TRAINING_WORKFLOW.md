# 🎓 Beaver ARS 학습 워크플로우

## 📋 전체 프로세스

```
1. 데이터 준비 → 2. Intent 분류 학습 → 3. NER 모델 학습 → 4. 모델 검증 → 5. 배포
```

---

## 1️⃣ 데이터 준비

### Intent Classification 데이터

**위치**: `data/sample/intent_sample.csv`

**형식**:
```csv
text,label
김치찌개 2개 주문할게요,order_food
영업시간이 언제인가요?,inquiry_hours
배달 가능한가요?,inquiry_delivery
```

**필요한 데이터**:
- 최소 **48개 클래스 × 100개 샘플** = 4,800개
- 권장: 클래스당 200-500개 (총 10,000~24,000개)

### NER 데이터

**위치**: `data/sample/ner_sample.conll`

**형식** (CoNLL):
```
김치찌개	B-FOOD
2개	B-QUANTITY
주문	O
할게요	O

불고기	B-FOOD
3인분	B-QUANTITY
```

**엔티티 타입** (6개):
- `FOOD` - 음식명
- `QUANTITY` - 수량
- `TIME` - 시간
- `OPTION` - 옵션 (맵기, 크기 등)
- `LOCATION` - 장소
- `PERSON` - 인명

---

## 2️⃣ Intent Classification 학습

### 단계별 실행

```bash
# 1. 데이터 준비 확인
python scripts/data_preprocessing.py --input data/raw/intent_data.csv --output data/processed/

# 2. Intent 매핑 확인
python src/241215_step0_intent_mapping.py

# 3. 학습 실행
python src/241215_step1_train_cls_intent.py \
    --data_path data/processed/intent_train.csv \
    --output_dir models/intent_classifier \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 2e-5

# 4. 평가
python src/241215_step1_evaluation_cls_intent.py \
    --model_path models/intent_classifier/best_model.pt \
    --test_data data/processed/intent_test.csv
```

### 예상 소요 시간

| 환경 | 데이터 크기 | 소요 시간 |
|------|------------|-----------|
| **GPU (RTX 3090)** | 10K samples | ~2시간 |
| **GPU (T4)** | 10K samples | ~4시간 |
| **CPU** | 10K samples | ~8-12시간 |

### 학습 모니터링

```bash
# WandB 대시보드에서 실시간 확인
# - Loss curve
# - Accuracy
# - Learning rate
# - Confusion matrix
```

---

## 3️⃣ NER 모델 학습

### 단계별 실행

```bash
# 1. NER 데이터 준비
python scripts/data_preprocessing.py --task ner \
    --input data/raw/ner_data.conll \
    --output data/processed/

# 2. 학습 실행
python src/241218_step1_ner_train_i_tagging.py \
    --data_path data/processed/ner_train.conll \
    --output_dir models/ner_model \
    --num_epochs 15 \
    --batch_size 16 \
    --learning_rate 2e-5

# 3. 평가
python src/241218_step2_ner_evaluation.py \
    --model_path models/ner_model/best_model.pt \
    --test_data data/processed/ner_test.conll
```

### 예상 소요 시간

| 환경 | 데이터 크기 | 소요 시간 |
|------|------------|-----------|
| **GPU (RTX 3090)** | 5K samples | ~1.5시간 |
| **GPU (T4)** | 5K samples | ~3시간 |
| **CPU** | 5K samples | ~6-10시간 |

---

## 4️⃣ 모델 검증

### Intent Classifier 검증

```bash
# 추론 테스트
python src/241215_step1_inference_cls_intent.py \
    --model_path models/intent_classifier/best_model.pt \
    --text "김치찌개 2개 주문할게요"

# 예상 출력:
# Intent: order_food
# Confidence: 0.987
```

### NER 모델 검증

```bash
# 엔티티 추출 테스트
python src/241218_step2_ner_evaluation.py \
    --model_path models/ner_model/best_model.pt \
    --text "김치찌개 2개 주문할게요"

# 예상 출력:
# Entities:
# - FOOD: 김치찌개
# - QUANTITY: 2개
```

### 성능 목표

| 메트릭 | Intent | NER |
|--------|--------|-----|
| **정확도/F1** | > 95% | > 93% |
| **추론 속도** | < 50ms | < 100ms |

---

## 5️⃣ 통합 테스트

### 전체 시스템 테스트

```bash
# 1. Intent + NER + Search 통합
python src/main_system.py \
    --intent_model models/intent_classifier/best_model.pt \
    --ner_model models/ner_model/best_model.pt \
    --query "김치찌개 2개 주문할게요"

# 2. API 서버 테스트
python src/web_server.py

# 3. 다른 터미널에서 요청
curl -X POST http://localhost:5000/order \
  -H "Content-Type: application/json" \
  -d '{"user_message": "김치찌개 2개 주문할게요"}'
```

---

## 📊 학습 체크리스트

### Intent Classification

- [ ] 데이터 준비 완료 (최소 4,800개)
- [ ] Train/Val/Test split (70/15/15)
- [ ] 클래스 불균형 확인
- [ ] 학습 실행 (20 epochs)
- [ ] 검증 정확도 > 95%
- [ ] Confusion matrix 확인
- [ ] 모델 저장 (`models/intent_classifier/`)

### NER

- [ ] CoNLL 형식 데이터 준비
- [ ] 엔티티 라벨 확인 (6 types)
- [ ] Train/Val/Test split
- [ ] 학습 실행 (15 epochs)
- [ ] F1-score > 93%
- [ ] Entity-level 평가
- [ ] 모델 저장 (`models/ner_model/`)

### 통합 테스트

- [ ] Intent + NER 동시 추론 테스트
- [ ] Hybrid search 테스트
- [ ] API 엔드포인트 테스트
- [ ] 부하 테스트 (100 requests/sec)
- [ ] 메모리 사용량 확인

---

## 🔧 트러블슈팅

### OOM (Out of Memory) 에러

```python
# batch_size 줄이기
--batch_size 8  # 16 → 8

# gradient accumulation 사용
--gradient_accumulation_steps 2
```

### 학습이 너무 느린 경우

```python
# Mixed precision 사용
--fp16

# Smaller model 시도
--model_name klue/bert-base  # roberta-large 대신
```

### Overfitting

```python
# Dropout 증가
--dropout 0.3

# Weight decay 증가
--weight_decay 0.01

# Early stopping
--early_stopping_patience 3
```

---

## 📁 학습 후 폴더 구조

```
models/
├── intent_classifier/
│   ├── best_model.pt           # 학습된 모델
│   ├── config.json             # 모델 설정
│   ├── tokenizer/              # 토크나이저
│   ├── training_args.json      # 학습 파라미터
│   └── metrics.json            # 성능 메트릭
│
└── ner_model/
    ├── best_model.pt
    ├── config.json
    ├── label_map.json          # 라벨 매핑
    ├── tokenizer/
    └── metrics.json
```

---

## ⏱️ 전체 학습 타임라인

### GPU 환경 (RTX 3090)

```
Day 1 (4시간):
├── 데이터 준비 (1시간)
├── Intent 학습 (2시간)
└── Intent 검증 (1시간)

Day 2 (3시간):
├── NER 학습 (1.5시간)
├── NER 검증 (0.5시간)
└── 통합 테스트 (1시간)
```

### CPU 환경

```
Week 1:
├── 데이터 준비 (1일)
├── Intent 학습 (2-3일)
└── Intent 검증 (1일)

Week 2:
├── NER 학습 (2일)
├── NER 검증 (1일)
└── 통합 테스트 (1일)
```

---

## 🚀 Quick Start (빠른 시작)

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 샘플 데이터로 빠른 테스트 (GPU 필요)
# Intent (5분)
python src/241215_step1_train_cls_intent.py \
    --data_path data/sample/intent_sample.csv \
    --num_epochs 3 \
    --quick_test

# NER (5분)
python src/241218_step1_ner_train_i_tagging.py \
    --data_path data/sample/ner_sample.conll \
    --num_epochs 3 \
    --quick_test

# 4. 추론 테스트
python src/241215_step1_inference_cls_intent.py \
    --text "김치찌개 주문할게요"
```

---

## 📞 다음 단계

학습이 완료되면:

1. **모델 평가**: `scripts/model_evaluation.py`
2. **모델 Export**: `scripts/export_model.py` (ONNX/TorchScript)
3. **배포**: `docker-compose up -d`
4. **모니터링**: Grafana 대시보드 확인

---

**💡 Tip**: WandB 계정을 만들어서 학습 과정을 시각화하면 더 편리합니다!

```bash
wandb login
# API 키 입력
```
