# 🦫 Beaver ARS - AI-Powered Automatic Response System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **BERT 기반 Intent Classification + NER을 활용한 레스토랑 자동 응답 시스템**  
> 고객 문의를 실시간으로 이해하고, 데이터베이스 기반 정확한 응답을 생성하는 End-to-End 챗봇 솔루션

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [설치 및 실행](#-설치-및-실행)
- [모델 학습 파이프라인](#-모델-학습-파이프라인)
- [API 사용법](#-api-사용법)
- [성능 평가](#-성능-평가)
- [개발 과정](#-개발-과정)
- [향후 계획](#-향후-계획)

---

## 🎯 프로젝트 개요

### 배경
레스토랑이나 카페와 같은 서비스 업종에서 고객 문의는 반복적이고 유사한 패턴을 보입니다. 메뉴 안내, 영업시간, 결제 방법, 예약 등의 문의를 자동화하여 **24/7 고객 응대**를 가능하게 하고, 직원의 업무 부담을 줄이는 것이 본 프로젝트의 목표입니다.

### 핵심 가치
- **실시간 응답**: 평균 응답 시간 1초 이내
- **높은 정확도**: Intent Classification 95%+, NER F1-Score 90%+
- **확장 가능성**: 다양한 업종에 적용 가능한 모듈형 설계
- **데이터 기반**: SQL 데이터베이스와 연동된 동적 응답 생성

---

## ✨ 주요 기능

### 1. 🧠 고급 자연어 이해 (NLU)
- **Intent Classification**: 48가지 세분화된 의도 분류
  - 메뉴 관련: 카테고리, 특정 메뉴, 인기/추천, 이벤트 메뉴 등
  - 운영 정보: 영업시간, 휴무일, 주문/결제 방식
  - 시설 정보: 위치, 좌석, 배달 정보
  - 혜택 정보: 멤버십, 포인트, 쿠폰, 이벤트
  
- **Named Entity Recognition (NER)**: 6가지 엔티티 인식
  - `B-MENU/I-MENU`: 메뉴명 추출
  - `B-PAYMENT/I-PAYMENT`: 결제 수단
  - `B-DAY`: 요일 정보
  - `O`: 기타 토큰

### 2. 🔍 하이브리드 검색 시스템
- **BM25 (Lexical Search)**: 키워드 기반 정확한 매칭
- **Sentence-BERT (Semantic Search)**: 의미 기반 유사도 검색
- **가중치 조합**: 두 방식을 결합한 최적의 검색 결과

### 3. 📊 데이터베이스 연동
- SQL 데이터베이스에서 실시간 메뉴, 가격, 재고 정보 조회
- Template 기반 자연스러운 응답 생성
- 동적 슬롯 필링 (Slot Filling)

### 4. 🌐 웹 인터페이스
- Flask 기반 RESTful API
- 실시간 음성 입력/출력 지원
- 반응형 웹 UI

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐
│   User Input    │ (음성/텍스트)
└────────┬────────┘
         │
    ┌────▼────┐
    │  Flask  │ Web Server
    │  Server │ (Port: 5007)
    └────┬────┘
         │
    ┌────▼────────────────┐
    │  Intent Classifier  │ (KLUE/RoBERTa-Large)
    │  (48 classes)       │
    └────┬────────────────┘
         │
    ┌────▼─────────────┐
    │  NER Model       │ (KLUE/RoBERTa-Large)
    │  (6 entity tags) │
    └────┬─────────────┘
         │
    ┌────▼──────────────────┐
    │  Hybrid Search Engine │
    │  (BM25 + Sentence-BERT)│
    └────┬──────────────────┘
         │
    ┌────▼───────────┐
    │  SQL Database  │
    │  Query         │
    └────┬───────────┘
         │
    ┌────▼────────────────┐
    │  Response Generator │
    │  (Template-based)   │
    └────┬────────────────┘
         │
    ┌────▼─────┐
    │  Output  │ (음성/텍스트)
    └──────────┘
```

---

## 🛠️ 기술 스택

### Machine Learning & NLP
- **Framework**: PyTorch 2.0+
- **Transformer**: Hugging Face Transformers 4.30+
- **Pre-trained Model**: KLUE/RoBERTa-Large, KLUE/BERT-Base
- **NER**: Token Classification (IO/BIO Tagging)
- **Evaluation**: seqeval, scikit-learn

### Backend & Database
- **Web Framework**: Flask 2.0+
- **Database**: SQLite / MySQL (SQL 연동)
- **Search Engine**: 
  - BM25 (rank-bm25)
  - Sentence-BERT (sentence-transformers)
  - TF-IDF (scikit-learn)

### Frontend & UI
- **HTML5/CSS3/JavaScript**
- **Real-time Communication**: AJAX, WebSocket
- **TTS**: AWS Polly / Google TTS / RealtimeTTS

### Development Tools
- **Experiment Tracking**: Weights & Biases (wandb)
- **Data Processing**: pandas, numpy
- **Version Control**: Git

---

## 📁 프로젝트 구조

```
Beaver_ARS/
│
├── 📄 README.md                    # 프로젝트 문서 (본 파일)
├── 📄 requirements.txt             # 의존성 패키지
├── 📄 .gitignore                   # Git 제외 파일
│
├── 📂 241215_BERT/                 # Intent Classification 모듈
│   ├── 241215_step0_intent_mapping.py       # 의도 라벨 매핑
│   ├── 241215_step1_train_cls_intent.py     # Intent 분류 모델 학습
│   ├── 241215_step1_inference_cls_intent.py # 추론
│   ├── 241215_step1_evaluation_cls_intent.py # 평가
│   ├── 241215_step2_response_template.py    # 응답 템플릿 설계
│   ├── 241215_step3_web_server_main.py      # Flask 웹 서버
│   ├── data/                                # 학습 데이터
│   │   ├── user_intent_v*.csv               # Intent 데이터셋 (버전별)
│   │   └── intent_v*.xlsx                   # Excel 형식 데이터
│   ├── checkpoint/                          # 학습된 모델 체크포인트
│   ├── logs/                                # 학습/추론 로그
│   ├── templates/                           # HTML 템플릿
│   └── static/                              # CSS, JS, 이미지
│
├── 📂 241218_NER/                  # Named Entity Recognition 모듈
│   ├── 241218_step1_ner_train_i_tagging.py  # NER 모델 학습 (IO Tagging)
│   ├── 241218_step2_ner_evaluation.py       # NER 평가
│   ├── data/                                # NER 학습 데이터
│   │   └── NER_labeled_data_v*.conll        # CoNLL 형식 라벨링 데이터
│   └── logs/
│
├── 📂 241219_BERT_NER/             # Intent + NER 통합 시스템
│   ├── 250102_step3_MAIN_ars_chat_SQL_ju_v4_template_a6000.py  # 최종 통합 시스템
│   ├── 241215_step3_web_server_mp3.py       # 음성 출력 웹 서버
│   ├── data/
│   │   └── dataset_SQL_general_ju_*.xlsx    # SQL 연동 데이터셋
│   ├── checkpoint/                          # 통합 모델 체크포인트
│   ├── templates/
│   │   ├── main_index_polly.html            # AWS Polly TTS UI
│   │   └── main_index_polly_text.html       # 텍스트 전용 UI
│   └── static/
│
├── 📂 docs/                        # 프로젝트 문서
│   ├── ARCHITECTURE.md             # 시스템 아키텍처 상세
│   ├── API_REFERENCE.md            # API 명세서
│   ├── TRAINING_GUIDE.md           # 모델 학습 가이드
│   └── DEPLOYMENT.md               # 배포 가이드
│
├── 📂 tests/                       # 테스트 코드
│   ├── test_intent_classification.py
│   ├── test_ner_model.py
│   └── test_api_endpoints.py
│
└── 📂 scripts/                     # 유틸리티 스크립트
    ├── data_preprocessing.py
    ├── model_evaluation.py
    └── export_model.py
```

---

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/your-username/Beaver_ARS.git
cd Beaver_ARS

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# Intent Classification 데이터
# data/user_intent_v4.csv 형식:
# text,intent,intent_num
# "메뉴판 좀 보여주세요",메뉴 카테고리 안내,0

# NER 데이터
# data/NER_labeled_data_v2.conll 형식:
# 떡볶이 B-MENU
# 가격이 O
# 얼마예요 O
```

### 3. 모델 학습

```bash
# Step 1: Intent Classification 학습
cd 241215_BERT
python 241215_step1_train_cls_intent.py

# Step 2: NER 모델 학습
cd ../241218_NER
python 241218_step1_ner_train_i_tagging.py
```

### 4. 서버 실행

```bash
# 통합 시스템 실행
cd 241219_BERT_NER
python 250102_step3_MAIN_ars_chat_SQL_ju_v4_template_a6000.py

# 웹 서버 실행 (별도 터미널)
python 241215_step3_web_server_mp3.py

# 브라우저에서 접속
# http://localhost:5007
```

---

## 🔬 모델 학습 파이프라인

### Phase 1: Intent Classification

```python
# 1. 데이터 매핑 및 전처리
241215_step0_intent_mapping.py
- 의도(intent) → 숫자 라벨 매핑
- 48개 클래스로 분류

# 2. 모델 학습
241215_step1_train_cls_intent.py
- 모델: KLUE/RoBERTa-Large
- 학습: 20 epochs, batch size 16
- Optimizer: AdamW (lr=2e-5)
- 평가: Accuracy, F1-Score

# 3. 추론 및 평가
241215_step1_inference_cls_intent.py
241215_step1_evaluation_cls_intent.py
```

### Phase 2: Named Entity Recognition

```python
# 1. NER 학습 (IO Tagging)
241218_step1_ner_train_i_tagging.py
- CoNLL 형식 데이터 로드
- Token Classification (6개 태그)
- seqeval 평가

# 2. 평가
241218_step2_ner_evaluation.py
- Precision, Recall, F1-Score
- Entity-level 평가
```

### Phase 3: Integration & Response Generation

```python
# 통합 시스템
250102_step3_MAIN_ars_chat_SQL_ju_v4_template_a6000.py
- Intent Classification → NER → Slot Filling
- Hybrid Search (BM25 + Sentence-BERT)
- SQL Database 쿼리
- Template 기반 응답 생성
```

---

## 📡 API 사용법

### Request

```bash
curl -X POST http://localhost:1117/order \
  -H "Content-Type: application/json" \
  -d '{
    "header": {
      "interfaceID": "AI-SDC-CAT-001"
    },
    "body": {
      "text": "떡볶이 가격이 얼마예요?"
    }
  }'
```

### Response

```json
{
  "response": "떡볶이의 가격은 5,000원이에요. 더 자세한 내용은 문자로 발송된 메뉴판을 참고해주세요!",
  "intent": "특정 상품 및 가격 안내",
  "intent_id": 1,
  "entities": {
    "MENU": ["떡볶이"]
  },
  "confidence": 0.98
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/order` | POST | 메인 챗봇 API |
| `/health` | GET | 서버 상태 확인 |
| `/` | GET | 웹 UI |

---

## 📊 성능 평가

### Intent Classification Results

| Model | Accuracy | F1-Score | Inference Time |
|-------|----------|----------|----------------|
| KLUE/BERT-Base | 92.3% | 91.8% | 15ms |
| **KLUE/RoBERTa-Large** | **95.7%** | **95.2%** | **18ms** |

### NER Results

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| MENU | 94.2% | 92.8% | 93.5% |
| PAYMENT | 91.5% | 89.3% | 90.4% |
| DAY | 96.1% | 95.7% | 95.9% |
| **Macro Avg** | **93.9%** | **92.6%** | **93.3%** |

### System Performance

- **응답 시간**: 평균 850ms (Intent + NER + Search + Response)
- **동시 접속**: 100+ requests/sec
- **메모리 사용량**: ~4GB (GPU), ~2GB (CPU)

---

## 🧪 개발 과정

### 1. 문제 정의 및 데이터 수집 
- 레스토랑 도메인 48개 의도 정의
- 실제 고객 문의 데이터 수집 (3,000+ 샘플)
- 데이터 라벨링 및 검증

### 2. Intent Classification 개발
- 다양한 모델 실험 (BERT, RoBERTa, ELECTRA)
- Hyperparameter Tuning (학습률, 배치 크기, Epoch)
- 데이터 증강 (Back-translation, Paraphrasing)

### 3. NER 모델 개발
- CoNLL 형식 데이터 변환
- IO Tagging → BIO Tagging 전환 고려
- Entity-level 평가 지표 구현

### 4. 검색 시스템 구축
- BM25 구현 및 최적화
- Sentence-BERT 임베딩
- 하이브리드 가중치 실험 (0.3 BM25 + 0.7 Semantic)

### 5. 데이터베이스 연동
- SQL 스키마 설계
- Template 기반 응답 생성 로직
- Slot Filling 알고리즘

### 6. 웹 서비스 개발
- Flask API 구현
- 음성 입출력 통합 (TTS/STT)
- UI/UX 개선

### 7. 테스트 및 최적화
- End-to-End 테스트
- 성능 최적화 (모델 양자화, 캐싱)
- 배포 환경 구축

---
