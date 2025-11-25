# 🦫 Beaver ARS - AI-Powered Automatic Response System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.57+-yellow.svg)](https://huggingface.co/transformers/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **KLUE/RoBERTa-Large 기반 Intent Classification (48 classes) + NER (6 entities)을 활용한 레스토랑 자동 응답 시스템**  
> 고객 문의를 실시간으로 이해하고, 하이브리드 검색 + SQL DB 기반 정확한 응답을 생성하는 프로덕션급 챗봇 솔루션

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [핵심 성과](#-핵심-성과)
- [시스템 아키텍처](#-시스템-아키텍처)
- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [빠른 시작](#-빠른-시작)
- [모델 학습](#-모델-학습)
- [배포](#-배포)
- [API 사용법](#-api-사용법)
- [성능 평가](#-성능-평가)

---

## 🎯 프로젝트 개요

레스토랑 및 서비스 업종의 반복적인 고객 문의(메뉴, 영업시간, 결제 방법 등)를 AI로 자동화하여 **24/7 고객 응대**를 실현하는 엔터프라이즈급 챗봇 시스템입니다.

### 🎯 핵심 목표
- **높은 정확도**: Intent 95.7%, NER 99.4% F1-Score
- **빠른 응답**: 평균 800ms 이내 응답 생성
- **프로덕션 준비**: Docker + Nginx + Prometheus 모니터링
- **확장 가능**: 48개 Intent, 다양한 도메인 적용 가능

---

## 🏆 핵심 성과

| 항목 | 성능 | 비고 |
|------|------|------|
| **Intent Classification** | **Accuracy 91.2%** | KLUE/RoBERTa-Large |
| **NER Model** | **F1-Score 99.5%** | Macro-averaged |
| **응답 시간** | **< 1초** | Intent + NER + Search + DB |
| **데이터셋** | **2,824 samples** | 48 intents, 6 entity types |
| **동시 처리** | **병렬 처리 지원** | Gunicorn multi-worker |

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│              (Web UI / REST API / WebSocket)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │      Nginx (Reverse Proxy)  │
            │  - Rate Limiting (100 req/m)│
            │  - SSL/TLS Termination      │
            │  - Load Balancing           │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │    Flask Application Server │
            │  (Gunicorn 4 workers)       │
            └──────────────┬──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌──────▼──────┐   ┌──────▼──────┐
   │ Intent  │      │ NER Model   │   │   Redis     │
   │Classifier│     │(Token Class)│   │   Cache     │
   │ 48 cls  │      │  6 entities │   │             │
   └────┬────┘      └──────┬──────┘   └─────────────┘
        │                  │
        └──────────┬───────┘
                   │
        ┌──────────▼──────────────┐
        │   Hybrid Search Engine  │
        │  - BM25 (40%)           │
        │  - Sentence-BERT (60%)  │
        │  - Cosine Similarity    │
        └──────────┬──────────────┘
                   │
        ┌──────────▼──────────────┐
        │   SQL Database (MySQL)  │
        │  - Menus Table          │
        │  - Business Info        │
        │  - Conversation Logs    │
        └──────────┬──────────────┘
                   │
        ┌──────────▼──────────────┐
        │   Response Generator    │
        │  - Slot Filling         │
        │  - Template Engine      │
        │  - TTS Integration      │
        └──────────┬──────────────┘
                   │
                   ▼
            ┌─────────────┐
            │   Output    │
            │ (JSON/Text) │
            └─────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Monitoring Stack                          │
│  Prometheus → Metrics Collection                            │
│  Grafana → Real-time Dashboards                             │
│  Node Exporter → System Metrics                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ 주요 기능

### 1. 🧠 정밀한 자연어 이해 (NLU)

**Intent Classification** - 48개 세분화된 의도 분류
```
📌 메뉴 (11개): 카테고리, 가격, 옵션, 인기/추천, 할인, 신메뉴 등
📌 주문/결제 (5개): 주문 방식, 결제 수단, 개별 결제 등
📌 영업 정보 (6개): 영업시간, 휴무일, 브레이크 타임, 특정 요일 등
📌 배달 (3개): 배달 지역, 배달비, 최소 주문 금액
📌 시설 (7개): 주소, 주차, 좌석, 단체석, 테라스 등
📌 혜택 (8개): 멤버십, 포인트, 쿠폰, 이벤트 등
📌 기타 (8개): 감사, 인사, Fallback 등
```

**Named Entity Recognition** - 6개 엔티티 추출
```python
{
  "MENU": ["떡볶이", "김밥"],        # B-MENU, I-MENU
  "PAYMENT": ["카드", "제로페이"],   # B-PAYMENT, I-PAYMENT
  "DAY": ["월요일"]                 # B-DAY
}
```

### 2. 🔍 하이브리드 검색 엔진

```python
# 키워드 + 의미 유사도 결합
final_score = 0.4 * BM25_score + 0.6 * SBERT_score
```

- **BM25**: 키워드 기반 정확 매칭 (Lexical)
- **Sentence-BERT**: 의미 기반 유사도 (Semantic)
- **TF-IDF**: 보조 검색 메커니즘

### 3. 📊 SQL 데이터베이스 연동

- **동적 응답 생성**: DB에서 실시간 메뉴/가격 조회
- **Slot Filling**: NER 추출 엔티티를 응답 템플릿에 삽입
- **MySQL/SQLite 지원**: 프로덕션/개발 환경 분리

### 4. 🚀 프로덕션급 인프라

- **Docker 컨테이너화**: 일관된 배포 환경
- **Nginx 리버스 프록시**: Rate limiting, SSL/TLS
- **Prometheus + Grafana**: 실시간 모니터링
- **CI/CD**: GitHub Actions 자동 배포

---

## 🛠️ 기술 스택

### 🤖 Machine Learning & NLP
| Category | Technology | Version |
|----------|------------|---------|
| **Framework** | PyTorch | 2.7.0 (CUDA 12.4) |
| **Transformers** | Hugging Face | 4.57.1 |
| **Pre-trained Model** | KLUE/RoBERTa-Large | 1024 hidden, 24 layers |
| **Search** | Sentence-BERT | paraphrase-multilingual-MiniLM-L12-v2 |
| **Evaluation** | seqeval, scikit-learn | 1.2.2, 1.2.0 |
| **Experiment Tracking** | W&B | 0.15+ |

### 🌐 Backend & Infrastructure
| Category | Technology | Purpose |
|----------|------------|---------|
| **Web Server** | Flask 2.0+ | REST API |
| **WSGI** | Gunicorn | Production server (4 workers) |
| **Database** | MySQL 8.0 | Data storage |
| **Cache** | Redis 7.0 | Response caching |
| **Proxy** | Nginx | Rate limiting, SSL/TLS |
| **Container** | Docker 24.0+ | Containerization |

### 📊 Monitoring & DevOps
- **Metrics**: Prometheus (time-series metrics)
- **Visualization**: Grafana (real-time dashboards)
- **CI/CD**: GitHub Actions
- **Logging**: Python logging module

### 🔍 Search Engine Components
- **BM25**: Keyword-based exact matching (rank-bm25)
- **Sentence-BERT**: Semantic similarity
- **TF-IDF**: Fallback search mechanism

### 💻 Hardware Requirements
- **Training**: NVIDIA RTX 5090 32GB VRAM
- **Production**: 8+ CPU cores, 16GB RAM
- **Storage**: 50GB+ (models + data + logs)

---

## 📁 프로젝트 구조

```
Beaver_ARS/
│
├── 📄 README.md                    # 프로젝트 문서
├── 📄 DATA_GUIDE.md                # 데이터셋 가이드
├── 📄 requirements.txt             # Python 의존성
├── 📄 train.sh                     # 자동화된 학습 스크립트
├── 📄 train_quick.py               # 빠른 학습 스크립트
├── 📄 docker-compose.yml           # 멀티 컨테이너 설정
├── 📄 Dockerfile                   # 컨테이너 이미지 정의
├── 📄 gunicorn_config.py           # WSGI 서버 설정
├── 📄 deploy.sh                    # 프로덕션 배포 스크립트
├── 📄 rollback.sh                  # 롤백 스크립트
│
├── 📂 data/                        # 데이터셋
│   ├── train/
│   │   ├── intent_data.csv         # Intent 학습 데이터 (2,824 samples)
│   │   └── ner_data.conll          # NER 학습 데이터 (CoNLL format)
│   ├── sample/                     # 샘플 데이터
│   └── processed/                  # 전처리된 데이터 (자동 생성)
│
├── 📂 src/                         # 소스 코드
│   ├── 241215_step0_intent_mapping.py       # Intent 라벨 매핑
│   ├── 241215_step1_train_cls_intent.py     # Intent 분류 학습
│   ├── 241215_step1_inference_cls_intent.py # Intent 추론
│   ├── 241215_step1_evaluation_cls_intent.py # Intent 평가
│   ├── 241218_step1_ner_train_i_tagging.py  # NER 학습
│   ├── 241218_step2_ner_evaluation.py       # NER 평가
│   ├── 241215_step2_response_template.py    # 응답 템플릿
│   ├── main_ars_system.py          # 통합 ARS 시스템 (메인)
│   ├── web_server.py               # Flask 웹 서버
│   └── main_system.py              # 시스템 엔트리포인트
│
├── 📂 models/                      # 학습된 모델
│   ├── intent_classifier/          # Intent 모델 체크포인트
│   └── ner_model/                  # NER 모델 체크포인트
│
├── 📂 templates/                   # HTML 템플릿
│   └── index.html                  # 웹 UI
│
├── 📂 static/                      # 정적 파일
│   ├── css/
│   ├── js/
│   └── audio/                      # TTS 오디오 파일
│
├── 📂 config/                      # 설정 파일
│   ├── config.yaml                 # 애플리케이션 설정
│   └── database.yaml               # 데이터베이스 설정
│
├── 📂 database/                    # DB 초기화
│   ├── init.sql                    # 스키마 및 초기 데이터
│   └── my.cnf                      # MySQL 설정
│
├── 📂 nginx/                       # Nginx 설정
│   └── nginx.conf                  # 리버스 프록시 설정
│
├── 📂 monitoring/                  # 모니터링
│   ├── prometheus.yml              # Prometheus 설정
│   └── grafana/                    # Grafana 대시보드
│
├── 📂 logs/                        # 로그 파일 (자동 생성)
│   └── YYYY-MM-DD/
│
├── 📂 tests/                       # 테스트 코드
│   ├── test_intent_classification.py
│   ├── test_ner_model.py
│   └── test_api_endpoints.py
│
├── 📂 scripts/                     # 유틸리티 스크립트
│   └── data_preprocessing.py
│
├── 📂 docs/                        # 상세 문서
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   └── DEPLOYMENT.md
│
└── 📂 .github/                     # CI/CD
    └── workflows/
        └── ci-cd.yml               # GitHub Actions
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/SynapseSmith/Beaver_ARS.git
cd Beaver_ARS

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

데이터는 `data/train/` 디렉토리에 준비되어 있습니다:
- `intent_data.csv`: 2,824개 Intent 샘플
- `ner_data.conll`: 2,170줄 NER 라벨링 데이터

상세한 데이터 구조는 [DATA_GUIDE.md](DATA_GUIDE.md)를 참조하세요.

### 3. 모델 학습 (선택사항)

사전 학습된 모델을 사용하거나, 직접 학습할 수 있습니다:

```bash
# 자동화된 학습 파이프라인 (권장)
chmod +x train.sh
./train.sh

# 또는 빠른 학습 (샘플 데이터)
python train_quick.py
```

### 4. 서버 실행

```bash
# Flask 개발 서버 실행
cd src
python web_server.py

# 브라우저에서 접속
# http://localhost:5000
```

### 5. API 테스트

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "떡볶이 가격이 얼마예요?"}'
```

---

## 🎓 모델 학습

### 자동화된 학습 파이프라인 (`train.sh`)

```bash
./train.sh
```

**7단계 자동 실행**:
1. ✅ Prerequisites 확인 (Python, GPU, packages)
2. ✅ 디렉토리 준비
3. ✅ 데이터 검증
4. ✅ Intent Classification 학습 (30 epochs)
5. ✅ Intent 모델 평가
6. ✅ NER 모델 학습 (30 epochs)
7. ✅ NER 모델 평가

### 최종 학습 파라미터

| 파라미터 | Intent | NER |
|----------|--------|-----|
| **Model** | KLUE/RoBERTa-Large | KLUE/RoBERTa-Large |
| **Epochs** | 30 | 30 |
| **Batch Size** | 16 | 8 |
| **Learning Rate** | 1e-4 | 2e-5 |
| **Gradient Accumulation** | 4 | 4 |
| **Effective Batch** | 64 | 32 |

### W&B 실험 추적

```bash
# W&B 로그인
wandb login

# 학습 중 실시간 모니터링
# https://wandb.ai/your-team/beaver-ars
```

---

## 🐳 배포

### Docker Compose (권장)

```bash
# 전체 스택 실행 (App + MySQL + Redis + Nginx + Prometheus + Grafana)
docker-compose up -d

# 로그 확인
docker-compose logs -f app

# 중지
docker-compose down
```

### 개별 Docker 빌드

```bash
# 이미지 빌드
docker build -t beaver-ars:latest .

# 컨테이너 실행
docker run -p 5000:5000 beaver-ars:latest
```

### 프로덕션 배포

```bash
# 배포 스크립트 실행
chmod +x deploy.sh
./deploy.sh

# 롤백 (문제 발생 시)
./rollback.sh
```

---

## 📡 API 사용법

### 챗봇 API

**Endpoint**: `POST /chat`

**Request**:
```json
{
  "text": "떡볶이 가격이 얼마예요?"
}
```

**Response**:
```json
{
  "response": "떡볶이의 가격은 5,000원입니다.",
  "intent": "특정 상품 및 가격 안내",
  "intent_id": 1,
  "confidence": 0.98,
  "entities": {
    "MENU": ["떡볶이"]
  },
  "response_time_ms": 850
}
```

### 헬스체크 API

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "db_connected": true,
  "uptime_seconds": 3600
}
```

### Prometheus 메트릭

**Endpoint**: `GET /metrics`

```
# 시스템 메트릭 수집
# - 요청 처리 시간
# - API 호출 카운트
# - 에러 발생률
# - 모델 로딩 상태
```

---

## 📊 성능 평가

### Intent Classification 결과

| Metric | Value |
|--------|-------|
| **Accuracy** | **91.2%** |
| **Weighted F1-Score** | **91.0%** |
| **Training Loss** | **0.509** |
| **Model** | **KLUE/RoBERTa-Large** |

**학습 데이터**:
- 총 2,824 샘플 (Train: 2,259 / Eval: 565)
- 48개 Intent 클래스
- 30 Epochs, Batch 16, LR 1e-4

### NER 결과

| Entity Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|------|
| **MENU** | 98.9% | 100.0% | 99.5% | 70 |
| **PAYMENT** | 100.0% | 100.0% | 100.0% | 5 |
| **DAY** | 100.0% | 100.0% | 100.0% | 17 |
| **Macro Avg** | **98.9%** | **100.0%** | **99.5%** | **92** |

### 시스템 성능

```
┌─────────────────────────┬────────────────────┐
│ 항목                     │ 성능               │
├─────────────────────────┼────────────────────┤
│ Intent 분류              │ 91.2% Accuracy     │
│ NER 추출                 │ 99.5% F1-Score     │
│ 학습 데이터              │ 2,824 samples      │
│ 지원 Intent              │ 48 classes         │
│ 지원 Entity              │ 6 types (MENU/DAY) │
│ GPU (학습)               │ RTX 5090 32GB      │
└─────────────────────────┴────────────────────┘
```

---

## 🔧 문제 해결 과정

### 1. NER 토큰 정렬 문제

**문제**: WordPiece 토크나이저 서브토큰 불일치  
**해결**: offset_mapping 기반 첫 서브토큰만 태깅  
**결과**: NER F1-Score 99.5% 달성

### 2. 하이브리드 검색 구현

**문제**: 단일 검색 방식의 한계  
**해결**: BM25(키워드) + SBERT(의미) 결합 (4:6 가중치)  
**결과**: 정확도와 유연성 모두 확보

### 3. GPU 메모리 최적화

**문제**: 대용량 모델 학습 시 메모리 부족  
**해결**: Gradient Accumulation (4 steps) 적용  
**결과**: Batch 16으로 Effective Batch 64 효과

### 4. 48개 Intent 세분화

**문제**: 포괄적 Intent로 정확도 저하  
**해결**: 메뉴/영업/배달/시설/혜택 등 48개로 세분화  
**결과**: 도메인별 정밀한 응답 생성 가능

---

</div>
