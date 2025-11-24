# 🚀 빠른 시작 가이드 (Quick Start)

프로젝트를 **최대한 빠르게** 실행하고 테스트하는 방법입니다.

---

## ⚡ 5분 안에 실행하기

### 1️⃣ 환경 설정 (1분)

```bash
# 프로젝트 폴더로 이동
cd Beaver_ARS_Portfolio

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ 데이터 준비 (선택사항)

**샘플 데이터로 바로 시작 가능!**

```bash
# 이미 준비된 샘플 데이터
ls data/sample/
# - intent_sample.csv  (Intent 분류 데이터)
# - ner_sample.conll   (NER 데이터)
```

**자신의 데이터 사용하려면:**
```bash
# Intent 데이터: CSV 형식
# text,label
# 김치찌개 주문할게요,order_food
# 영업시간이 언제인가요?,inquiry_hours

# NER 데이터: CoNLL 형식
# 김치찌개 B-FOOD
# 주문 O
# 할게요 O
```

### 3️⃣ 빠른 학습 (3분 - GPU 기준)

```bash
# 두 모델 모두 학습 (3 epoch - 테스트용)
python train_quick.py --train_all --epochs 3

# 또는 개별 학습
python train_quick.py --train_intent --epochs 3
python train_quick.py --train_ner --epochs 3
```

### 4️⃣ 추론 테스트 (10초)

```bash
# Intent 분류 테스트
python src/241215_step1_inference_cls_intent.py \
    --model_path models/intent_classifier/best_model.pt \
    --text "김치찌개 2개 주문할게요"

# 결과: Intent: order_food (Confidence: 0.95)
```

---

## 🎯 실제 학습 (프로덕션용)

### 방법 1: 자동화 스크립트

```bash
# 전체 학습 파이프라인 실행 (2-8시간)
chmod +x train.sh
./train.sh

# 스크립트가 자동으로:
# 1. 환경 체크
# 2. 데이터 검증
# 3. Intent 학습 (20 epochs)
# 4. Intent 평가
# 5. NER 학습 (15 epochs)
# 6. NER 평가
# 7. 추론 테스트
```

### 방법 2: 수동 실행

```bash
# 1. Intent Classifier 학습
python src/241215_step1_train_cls_intent.py \
    --data_path data/sample/intent_sample.csv \
    --output_dir models/intent_classifier \
    --num_epochs 20 \
    --batch_size 16

# 2. Intent 평가
python src/241215_step1_evaluation_cls_intent.py \
    --model_path models/intent_classifier/best_model.pt \
    --test_data data/sample/intent_sample.csv

# 3. NER 학습
python src/241218_step1_ner_train_i_tagging.py \
    --data_path data/sample/ner_sample.conll \
    --output_dir models/ner_model \
    --num_epochs 15 \
    --batch_size 16

# 4. NER 평가
python src/241218_step2_ner_evaluation.py \
    --model_path models/ner_model/best_model.pt \
    --test_data data/sample/ner_sample.conll
```

---

## 🐳 Docker로 실행 (학습 없이 API만)

```bash
# 1. 환경 변수 설정
cp .env.example .env

# 2. Docker Compose 실행
docker-compose up -d

# 3. API 테스트
curl -X POST http://localhost:5000/order \
  -H "Content-Type: application/json" \
  -d '{"user_message": "김치찌개 2개 주문할게요"}'

# 4. 대시보드 확인
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

---

## 📊 학습 모니터링

### WandB (추천)

```bash
# WandB 로그인
wandb login
# API 키 입력

# 학습 시작하면 자동으로 대시보드 생성됨
# - Loss curve
# - Accuracy
# - Confusion matrix
```

### 로컬 로그

```bash
# 학습 로그 확인
tail -f logs/intent_training.log
tail -f logs/ner_training.log

# TensorBoard (선택사항)
tensorboard --logdir logs/
```

---

## 🧪 테스트

### 1. 유닛 테스트

```bash
# 전체 테스트
pytest tests/ -v

# 특정 테스트
pytest tests/test_intent_classification.py -v
pytest tests/test_ner_model.py -v
pytest tests/test_api_endpoints.py -v
```

### 2. API 테스트

```bash
# API 서버 시작
python src/web_server.py &

# 요청 테스트
curl -X POST http://localhost:5000/order \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "김치찌개 2개 주문할게요"
  }'

# Health check
curl http://localhost:5000/health
```

### 3. 부하 테스트

```bash
# 100개 요청, 10개 동시 실행
python scripts/load_test.py --requests 100 --workers 10

# 결과:
# - Requests/sec: 45.2
# - Average response time: 187ms
# - 95th percentile: 320ms
```

---

## 🔧 GPU 설정

### CUDA 확인

```bash
# GPU 사용 가능 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# GPU 정보
nvidia-smi
```

### GPU 메모리 부족 시

```python
# batch_size 줄이기
--batch_size 8  # 기본 16 → 8

# Mixed precision 사용
--fp16

# Gradient accumulation
--gradient_accumulation_steps 2
```

---

## 📝 체크리스트

### 학습 전

- [ ] Python 3.8+ 설치 확인
- [ ] GPU 드라이버 설치 (선택)
- [ ] 가상환경 생성 및 활성화
- [ ] requirements.txt 설치
- [ ] 데이터 준비 (또는 샘플 데이터 사용)

### 학습 중

- [ ] GPU 사용률 확인 (`nvidia-smi`)
- [ ] 메모리 사용량 모니터링
- [ ] Loss가 감소하는지 확인
- [ ] WandB 대시보드 확인

### 학습 후

- [ ] 모델 파일 생성 확인 (`models/*/best_model.pt`)
- [ ] 평가 메트릭 확인 (Accuracy > 95%, F1 > 93%)
- [ ] 추론 테스트 성공
- [ ] API 테스트 성공

---

## 🆘 문제 해결

### ImportError: No module named 'torch'

```bash
pip install -r requirements.txt
```

### CUDA out of memory

```bash
# batch_size 줄이기
python train_quick.py --train_all --batch_size 8
```

### 데이터 파일을 찾을 수 없음

```bash
# 파일 경로 확인
ls -la data/sample/

# 절대 경로 사용
python train_quick.py --intent_data /full/path/to/data.csv
```

### 학습이 너무 느림 (CPU)

```bash
# Epoch 수 줄이기
python train_quick.py --train_all --epochs 3

# 또는 작은 모델 사용
# src/*.py 파일에서 model_name을 'klue/bert-base'로 변경
```

---

## 🚀 다음 단계

학습이 완료되면:

1. **성능 확인**: `logs/` 폴더의 평가 메트릭 확인
2. **추론 테스트**: 실제 문장으로 테스트
3. **API 서버 실행**: `python src/web_server.py`
4. **배포**: `docker-compose up -d`
5. **모니터링**: Grafana 대시보드 설정

---

## 📚 관련 문서

- [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) - 상세 학습 가이드
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - 학습 이론 및 팁
- [docs/PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md) - 배포 가이드
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 프로젝트 구조

---

## 💡 팁

### 빠른 프로토타이핑

```bash
# 매우 작은 데이터셋으로 1 epoch만 (1분)
python train_quick.py --train_all --epochs 1 --batch_size 8
```

### 최고 성능

```bash
# 더 많은 epoch, 큰 batch size (GPU 메모리 충분 시)
./train.sh
# 자동으로 20 epochs (Intent), 15 epochs (NER)
```

### 실험 추적

```bash
# WandB에 실험명 지정
export WANDB_PROJECT="beaver-ars"
export WANDB_NAME="experiment-v1"
python train_quick.py --train_all
```

---

**🎉 준비 완료! 학습을 시작해보세요!**

```bash
python train_quick.py --train_all --epochs 3
```
