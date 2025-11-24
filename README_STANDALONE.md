# Beaver ARS Portfolio - 독립 실행 가이드

이 프로젝트는 **Beaver_ARS에서 완전히 독립된** 자체 실행 가능한 ARS(자동 응답 시스템) 포트폴리오입니다.

## 📁 프로젝트 구조

```
Beaver_ARS_Portfolio/
├── src/                                    # 소스 코드
│   ├── 241215_step1_train_cls_intent.py    # Intent 분류 모델 학습
│   ├── 241215_step1_evaluation_cls_intent.py  # Intent 분류 평가
│   ├── 241215_step1_inference_cls_intent.py   # Intent 추론
│   ├── 241218_step1_ner_train_i_tagging.py    # NER 모델 학습
│   ├── 241218_step2_ner_evaluation.py         # NER 평가
│   ├── 241215_step3_web_server_mp3.py         # 음성 입력 웹 서버
│   ├── 241215_step3_web_server_text.py        # 텍스트 입력 웹 서버
│   └── main_ars_system.py                     # 메인 ARS 통합 시스템
├── data/                                   # 데이터 파일
│   ├── sample/                             # 샘플 데이터
│   │   ├── intent_sample.csv               # Intent 분류 샘플
│   │   └── ner_sample.conll                # NER 샘플
│   └── train/                              # 학습 데이터
│       ├── intent_data.csv                 # Intent 분류 데이터
│       └── ner_data.conll                  # NER 학습 데이터
├── models/                                 # 학습된 모델 저장
│   ├── intent_classifier/                  # Intent 분류 모델
│   └── ner_checkpoint/                     # NER 모델
├── logs/                                   # 실행 로그
├── static/                                 # 웹 UI 정적 파일
├── templates/                              # 웹 UI HTML 템플릿
├── config/                                 # 설정 파일
├── database/                               # SQLite DB
├── train.sh                                # 통합 학습 스크립트
└── requirements.txt                        # Python 패키지 의존성

```

## 🚀 빠른 시작

### 1. 가상환경 활성화

```bash
cd /opt/fastapi-poc
source venv/bin/activate
```

### 2. 의존성 설치 (최초 1회)

```bash
cd Beaver_ARS_Portfolio
pip install -r requirements.txt
```

### 3. 모델 학습

```bash
./train.sh
```

학습 스크립트는 다음 순서로 실행됩니다:
1. Intent 분류 모델 학습 (klue/bert-base)
2. Intent 분류 평가 및 성능 측정
3. NER 모델 학습 (Entity 추출)
4. NER 평가 및 성능 측정

### 4. 서비스 실행

#### 메인 ARS 시스템 (통합)
```bash
cd src
python main_ars_system.py
# http://localhost:5050 에서 실행
```

#### 텍스트 입력 웹 서버
```bash
cd src
python 241215_step3_web_server_text.py
# http://localhost:5000 에서 실행
```

#### 음성 입력 웹 서버
```bash
cd src
python 241215_step3_web_server_mp3.py
# http://localhost:5000 에서 실행 (음성 인식 지원)
```

## 📊 주요 기능

### 1. Intent Classification (의도 분류)
- **모델**: klue/bert-base
- **기능**: 사용자 발화의 의도를 50+ 카테고리로 분류
  - 메뉴 문의 (카테고리, 가격, 상세정보)
  - 결제 방법 안내
  - 영업시간/휴무일 안내
  - 배달/포장 관련 문의
  - 매장 위치/시설 안내
  - 멤버십/포인트/쿠폰 안내
  - 불만/칭찬/문의/예약
- **학습 데이터**: `data/train/intent_data.csv`
- **평가**: Accuracy, Precision, Recall, F1-Score

### 2. Named Entity Recognition (개체명 인식)
- **모델**: klue/bert-base + IOB Tagging
- **추출 Entity**:
  - B-MENU, I-MENU: 메뉴명
  - B-PAYMENT, I-PAYMENT: 결제 수단
  - B-DAY: 날짜/요일
- **학습 데이터**: `data/train/ner_data.conll` (CoNLL 포맷)
- **평가**: Entity-level Precision, Recall, F1 (seqeval)

### 3. 통합 ARS 시스템
- **구조**: Flask REST API
- **파이프라인**:
  1. 사용자 입력 수신 (텍스트/음성)
  2. Intent Classification
  3. NER Entity 추출
  4. 데이터베이스 조회 (메뉴/가격/영업시간)
  5. 응답 생성 및 반환
- **데이터베이스**: SQLite (주문/메뉴 관리)

## ⚙️ 메모리 최적화 설정

RTX 5090 (32GB) 환경에서 최적화된 설정:

```bash
# train.sh 에 포함된 설정
BATCH_SIZE=1                    # 배치 사이즈 최소화
GRADIENT_ACCUMULATION_STEPS=4   # Gradient 누적 (효과적 배치 = 4)
MODEL_NAME="klue/bert-base"     # 경량 모델 (roberta-large 대신)
FP16=true                       # 혼합 정밀도 학습
GRADIENT_CHECKPOINTING=true     # 메모리 절약 옵션
```

## 📝 스크립트 사용법

### Intent 모델 수동 학습
```bash
python src/241215_step1_train_cls_intent.py \
    --data_path data/train/intent_data.csv \
    --output_dir models/intent_classifier \
    --model_name klue/bert-base \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-5
```

### Intent 평가
```bash
python src/241215_step1_evaluation_cls_intent.py \
    --model_path models/intent_classifier \
    --test_data data/sample/intent_sample.csv \
    --output_xlsx logs/intent_evaluation.xlsx
```

### Intent 단일 추론
```bash
python src/241215_step1_inference_cls_intent.py \
    --model_path models/intent_classifier \
    --text "짜장면 얼마예요?"
```

### NER 모델 수동 학습
```bash
python src/241218_step1_ner_train_i_tagging.py \
    --data_path data/train/ner_data.conll \
    --output_dir models/ner_checkpoint \
    --model_name klue/bert-base \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-5
```

### NER 평가
```bash
python src/241218_step2_ner_evaluation.py \
    --model_path models/ner_checkpoint \
    --test_data data/sample/ner_sample.conll \
    --output_xlsx logs/ner_evaluation.xlsx
```

## 🔧 환경 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU 권장 (CUDA 지원)
  - 최소: 8GB VRAM
  - 권장: 24GB+ VRAM
- **RAM**: 16GB 이상
- **저장공간**: 10GB+ (모델 + 데이터)

### 소프트웨어
- **Python**: 3.8 이상
- **CUDA**: 11.8 이상 (GPU 사용 시)
- **OS**: Linux (Ubuntu 20.04+), Windows, macOS

### Python 패키지 (requirements.txt)
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
scikit-learn>=1.2.0
pandas>=2.0.0
seqeval>=1.2.2
openpyxl>=3.1.0
flask>=2.3.0
sentence-transformers>=2.2.0
accelerate>=0.20.0
```

## 📈 성능 지표

### Intent Classification (Example)
- **Accuracy**: ~92%
- **Macro F1-Score**: ~0.89
- **학습 시간**: ~30분 (3 epochs, RTX 5090)

### NER (Example)
- **Entity F1-Score**: ~0.85
- **Precision**: ~0.87
- **Recall**: ~0.83
- **학습 시간**: ~25분 (3 epochs, RTX 5090)

## 🐛 트러블슈팅

### 1. 메모리 부족 (std::bad_alloc)
```bash
# train.sh에서 BATCH_SIZE를 더 줄이기
BATCH_SIZE=1  # 이미 최소값
# 또는 더 작은 모델 사용
MODEL_NAME="klue/bert-base"  # 대신 distilbert 등
```

### 2. CUDA Out of Memory
```bash
# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# 또는 train.sh 수정하여 FP16 활성화 확인
```

### 3. 모듈을 찾을 수 없음 (ModuleNotFoundError)
```bash
# 가상환경 활성화 확인
source /opt/fastapi-poc/venv/bin/activate
# 의존성 재설치
pip install -r requirements.txt
```

### 4. 데이터 파일이 없음
```bash
# 샘플 데이터 확인
ls -lh data/sample/
ls -lh data/train/
# 필요 시 Beaver_ARS 폴더에서 복사
```

## 📞 지원 및 문의

- **프로젝트**: Beaver ARS (Automatic Response System)
- **목적**: 한국어 음성 주문 챗봇 시스템
- **기술 스택**: PyTorch, Transformers, Flask, BERT, NER
- **GPU**: NVIDIA GeForce RTX 5090 (32GB)

## 🔒 라이선스

이 프로젝트는 Portfolio 목적으로 작성되었습니다.

---

**참고**: 이 Portfolio는 Beaver_ARS 프로젝트에서 완전히 독립되어 있으며, 모든 경로는 `Beaver_ARS_Portfolio` 폴더 기준 상대 경로를 사용합니다.
