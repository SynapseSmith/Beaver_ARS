# Beaver ARS Source Code Documentation

## 📋 개요

Beaver ARS의 src/ 디렉터리는 인텐트 분류, NER, 응답 생성, 웹 서버 등 ARS 시스템의 핵심 컴포넌트들을 포함합니다.

## 🎯 표준 인텐트 체계 (48개)

모든 모듈은 동일한 48개 인텐트 클래스를 사용합니다:

| ID | 인텐트 | 설명 |
|----|--------|------|
| 0 | 메뉴 카테고리 안내 | 메뉴 카테고리 전반 안내 |
| 1 | 특정 상품 및 가격 안내 | 개별 상품 가격 문의 (슬롯 필요) |
| 2 | 상품에 대한 상세 및 추가 안내 | 상품 상세 정보 |
| 3 | (뱃지) 인기메뉴 | 베스트/인기 메뉴 |
| 4 | (뱃지) 추천메뉴 | 추천 메뉴 |
| 5 | (뱃지) 대표메뉴 | 대표 시그니처 메뉴 |
| 6 | (뱃지) 할인, 이벤트 | 할인 및 프로모션 |
| 7 | (뱃지) 신상 메뉴 | 신메뉴 |
| 8 | (뱃지) 한정 메뉴 | 기간 한정 메뉴 |
| 9 | (뱃지) 매운맛 | 매운맛 메뉴 |
| 10 | 주문 방식 안내 | 키오스크, 테이블오더, 스마트주문 등 |
| 11 | 주문한 상품 전달 방식 안내 | 내점, 포장, 배달 등 |
| 12 | 결제 방법 안내 | 결제 수단 전반 |
| 13 | 특정 결제 방법 상세 안내 | 특정 결제 수단 상세 (슬롯 필요) |
| 14 | 결제 방법 추가 안내 | 결제 관련 추가 정보 |
| 15 | 영업 시간 안내 | 기본 영업 시간 |
| 16 | 영업 시간 상세 안내 | 요일별 영업 시간 |
| 17 | 브레이크 타임 안내 | 브레이크 타임 기본 |
| 18 | 브레이크 타임 상세 안내 | 브레이크 타임 상세 |
| 19 | 휴무일 안내 | 정기/임시 휴무 |
| 20 | 휴무일 상세 안내 | 휴무일 상세 정보 |
| 21 | 영업시간 및 휴무일 추가 안내 | 영업 관련 추가 정보 |
| 22 | 특정 요일에 대한 영업 여부 | 특정 요일 영업 확인 (슬롯 필요) |
| 23 | 배달 가능 지역 안내 | 배달 가능 지역 |
| 24 | 배달비 및 최소 주문 금액 안내 | 배달비 정보 |
| 25 | 배달 추가 안내 | 배달 관련 추가 정보 |
| 26 | 테이블 점유 안내 | 테이블 현황 |
| 27 | 테이블 점유 추가 안내 | 테이블 관련 추가 정보 |
| 28 | 상점 주소 안내 및 지도 링크 전달 | 매장 위치 |
| 29 | 대중교통 이용 방법 안내 | 대중교통 안내 |
| 30 | 근처 랜드마크 안내 | 주변 랜드마크 |
| 31 | 테이블 및 좌석 수 안내 | 좌석 수 정보 |
| 32 | 야외 테라스 또는 개별 룸 여부 안내 | 특수 좌석 |
| 33 | (뱃지) 1+1 메뉴 문의 | 1+1 프로모션 메뉴 |
| 34 | 멤버십 가입 안내 | 회원 가입 방법 |
| 35 | 멤버십 혜택 안내 | 회원 혜택 |
| 36 | 포인트 적립 안내 | 포인트 적립 규칙 |
| 37 | 포인트 사용 안내 | 포인트 사용 방법 |
| 38 | 쿠폰 발행 안내 | 쿠폰 발급 |
| 39 | 쿠폰 사용 안내 | 쿠폰 사용 방법 |
| 40 | 멤버십 및 쿠폰에 대한 추가 안내 | 멤버십 관련 추가 정보 |
| 41 | 현재 진행 중인 이벤트 안내 | 이벤트 기본 |
| 42 | 현재 진행 중인 이벤트 상세 안내 | 이벤트 상세 |
| 43 | 현재 진행 중인 이벤트 추가 안내 | 이벤트 추가 정보 |
| 44 | CallBackIntent | 상담원 연결 |
| 45 | 감사 인텐트 | 감사 표현 |
| 46 | 인사 인텐트 | 인사 표현 |
| 47 | fallbackintent | 기타/미분류 |

> **중요**: ID 33은 "(뱃지) 1+1 메뉴 문의"이며, 이전 버전에서 잘못 배치되어 있던 것을 수정했습니다.

---

## 📁 파일 목록 및 설명

### 1️⃣ 메인 시스템

#### `main_ars_system.py` (프로덕션 API 서버) ⭐
- **목적**: Hybrid Search (BM25+SBERT) 기반 프로덕션 ARS API
- **기술 스택**: Flask, KLUE/RoBERTa-Large, KoE5 SBERT, BM25Okapi
- **주요 기능**:
  - 하이브리드 검색: `0.4 × BM25 + 0.6 × SBERT`
  - 인텐트 분류 (48 classes)
  - NER (6 entities: O, B-MENU, I-MENU, B-PAYMENT, I-PAYMENT, B-DAY)
- **엔드포인트**:
  - `GET /health` - 헬스 체크
  - `POST /predict` - 인텐트 분류
  - `POST /ner` - NER 추출
  - `POST /chat` - 대화형 응답 (하이브리드 검색)
  - `POST /order` - 주문 처리
  - `GET /intents` - 인텐트 목록 조회
  - `GET /metrics` - Prometheus 메트릭
- **사용법**:
  ```bash
  python src/main_ars_system.py
  # or
  gunicorn -c gunicorn_config.py src.main_ars_system:app
  ```

#### `main_system.py` (대체 시스템)
- **목적**: main_ars_system.py의 대체/백업 버전
- **특징**: 동일한 48개 인텐트 사용, 유사한 API 구조
- **사용 시나리오**: A/B 테스트, 백업 서버

#### `web_server.py` (간단한 웹 인터페이스)
- **목적**: 기본 Flask 웹 인터페이스
- **특징**: 최소한의 기능, RealtimeTTS 선택적 지원
- **사용법**: `python src/web_server.py`

#### `web_server_mp3.py` / `web_server_text.py`
- **목적**: MP3 음성 출력 / 텍스트 전용 웹 서버
- **특징**: 각각 음성/텍스트에 특화된 응답 방식

---

### 2️⃣ 인텐트 분류 모듈

#### `intent_training.py` (인텐트 모델 학습)
- **목적**: KLUE/RoBERTa-Large 기반 인텐트 분류 모델 학습
- **입력**: CSV 형식 학습 데이터 (user, intent_num)
- **출력**: 학습된 모델 (`models/intent_classifier/`)
- **주요 파라미터**:
  - `--data_path`: 학습 데이터 CSV 경로
  - `--output_dir`: 모델 저장 경로
  - `--model_name`: 기본 모델 (기본값: klue/roberta-large)
  - `--num_epochs`: 에포크 수 (기본값: 10)
  - `--batch_size`: 배치 크기 (기본값: 16)
  - `--learning_rate`: 학습률 (기본값: 2e-5)
- **사용법**:
  ```bash
  python src/intent_training.py \
      --data_path data/user_intent_v4.csv \
      --output_dir models/intent_classifier \
      --num_epochs 10 \
      --batch_size 16
  ```

#### `intent_evaluation.py` (인텐트 모델 평가)
- **목적**: 학습된 인텐트 모델 성능 평가
- **입력**: 테스트 CSV, 학습된 모델
- **출력**: Excel 평가 결과 (정확도, F1 스코어, 오분류 분석)
- **주요 파라미터**:
  - `--model_path`: 평가할 모델 경로
  - `--test_data`: 테스트 데이터 CSV
  - `--output_xlsx`: 결과 Excel 파일명
  - `--num_labels`: 인텐트 개수 (기본값: 48)
  - `--batch_size`: 배치 크기 (기본값: 16)
- **사용법**:
  ```bash
  python src/intent_evaluation.py \
      --model_path models/intent_classifier \
      --test_data data/test_intent.csv \
      --output_xlsx intent_test_results.xlsx
  ```
- **최근 수정**: ID 33 중복 제거 (올바른 순서: 0-47)

#### `intent_inference.py` (단일 텍스트 인텐트 추론)
- **목적**: 학습된 모델로 단일 텍스트 인텐트 분류
- **입력**: 텍스트 문자열
- **출력**: 예측 인텐트 ID 및 라벨
- **주요 파라미터**:
  - `--model_path`: 모델 경로
  - `--text`: 분류할 텍스트
- **사용법**:
  ```bash
  python src/intent_inference.py \
      --model_path models/intent_classifier \
      --text "짜장면 얼마에요?"
  ```

#### `intent_mapping.py` (데이터 전처리)
- **목적**: Excel 인텐트 데이터를 CSV로 변환 (텍스트 → 숫자 라벨)
- **입력**: `data/intent_v9.xlsx`
- **출력**: `data/user_intent_v4.csv` (user, intent_num)
- **기능**:
  - 인텐트 텍스트 → 숫자 매핑
  - 따옴표 제거
  - NaN 체크
  - 인텐트별 카운트 통계 (`intent_counts.xlsx`)
- **사용법**:
  ```bash
  python src/intent_mapping.py
  ```
- **최근 수정**: 24개 → 48개 인텐트로 업데이트

---

### 3️⃣ NER (Named Entity Recognition) 모듈

#### `ner_training.py` (NER 모델 학습)
- **목적**: KLUE/RoBERTa-Large 기반 NER 모델 학습
- **입력**: CoNLL 형식 데이터 (token\tBIO_label)
- **출력**: 학습된 NER 모델 (`models/ner_model/`)
- **엔티티**:
  - `O`: Outside (일반 토큰)
  - `B-MENU` / `I-MENU`: 메뉴명
  - `B-PAYMENT` / `I-PAYMENT`: 결제 수단
  - `B-DAY`: 요일/날짜
- **주요 파라미터**:
  - `--data_path`: CoNLL 데이터 경로
  - `--output_dir`: 모델 저장 경로
  - `--model_name`: 기본 모델 (기본값: klue/roberta-large)
  - `--num_epochs`: 에포크 수 (기본값: 15)
  - `--batch_size`: 배치 크기 (기본값: 8)
  - `--learning_rate`: 학습률 (기본값: 2e-5)
- **사용법**:
  ```bash
  python src/ner_training.py \
      --data_path data/ner_train.conll \
      --output_dir models/ner_model \
      --num_epochs 15
  ```

#### `ner_evaluation.py` (NER 모델 평가)
- **목적**: NER 모델 성능 평가
- **입력**: 테스트 CoNLL 데이터, 학습된 모델
- **출력**: Excel 평가 결과 (엔티티별 정밀도/재현율/F1)
- **주요 파라미터**:
  - `--model_path`: 평가할 모델 경로
  - `--test_data`: 테스트 CoNLL 데이터
  - `--output_xlsx`: 결과 Excel 파일명
- **사용법**:
  ```bash
  python src/ner_evaluation.py \
      --model_path models/ner_model \
      --test_data data/ner_test.conll \
      --output_xlsx ner_test_results.xlsx
  ```

---

### 4️⃣ 응답 생성

#### `response_templates.py` (템플릿 기반 응답 생성)
- **목적**: 인텐트별 응답 템플릿 관리 및 데이터베이스 쿼리
- **특징**:
  - 48개 인텐트용 응답 템플릿 (일부 구현 중)
  - Excel 데이터베이스 쿼리 (PandaSQL)
  - 슬롯 채우기 (STORE_NM, ROAD_NM_ADDR 등)
- **구현된 인텐트** (22개):
  - 0: 메뉴 카테고리
  - 1: 인기 메뉴 (QTY 기반)
  - 2: 배지 메뉴
  - 3: 주문 방식
  - 4: 결제 방법
  - 5: 영업 시간
  - 6: 휴무일
  - 7: 배달 지역
  - 8: 배달비
  - 9: 포장 여부
  - 10-12: 콜백 인텐트
  - 13: 주소
  - 14: 테이블 수
  - 15: 룸/야외 좌석
  - 16: 멤버십 가입
  - 17: 포인트 사용
  - 18: 쿠폰/스탬프
  - 19: 이벤트
  - 20-21: 콜백 인텐트
- **데이터베이스**: Excel 파일 기반 (상점 정보, 메뉴, 이벤트 등)
- **사용법**:
  ```python
  from response_templates import execute_sql, args
  result = execute_sql(intent_id=0, df=store_dataframe)
  template = args.response_templates[0][0].format(**result)
  ```

---

## 🔄 학습 파이프라인 워크플로우

### 인텐트 분류 모델 학습 전체 흐름

```
1. 데이터 준비
   └─> data/intent_v9.xlsx (Excel)

2. 데이터 전처리
   └─> python src/intent_mapping.py
       ├─> data/user_intent_v4.csv 생성
       └─> intent_counts.xlsx 통계

3. 모델 학습
   └─> python src/intent_training.py \
       --data_path data/user_intent_v4.csv \
       --output_dir models/intent_classifier
       └─> models/intent_classifier/ (저장)

4. 모델 평가
   └─> python src/intent_evaluation.py \
       --model_path models/intent_classifier \
       --test_data data/test_intent.csv
       └─> intent_test_results.xlsx

5. 프로덕션 배포
   └─> python src/main_ars_system.py
       └─> API 서버 실행 (포트 5000)
```

### NER 모델 학습 전체 흐름

```
1. 데이터 준비
   └─> data/ner_train.conll (CoNLL 형식)
       예시:
       짜장면    B-MENU
       주세요    O

2. 모델 학습
   └─> python src/ner_training.py \
       --data_path data/ner_train.conll \
       --output_dir models/ner_model
       └─> models/ner_model/ (저장)

3. 모델 평가
   └─> python src/ner_evaluation.py \
       --model_path models/ner_model \
       --test_data data/ner_test.conll
       └─> ner_test_results.xlsx

4. 프로덕션 통합
   └─> main_ars_system.py의 /ner 엔드포인트에서 사용
```

---

## 🔧 사용 예시

### 1. API 서버 실행 (Docker)

```bash
# docker-compose로 전체 시스템 실행
cd Beaver_ARS
docker-compose up -d

# 헬스 체크
curl http://localhost:5000/health

# 하이브리드 검색 채팅
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "짜장면 얼마에요?", "top_k": 5}'
```

### 2. 로컬 개발 환경

```bash
# 의존성 설치
pip install -r requirements.txt

# Flask 개발 서버
python src/main_ars_system.py

# Gunicorn 프로덕션 서버 (4 workers)
gunicorn -c gunicorn_config.py src.main_ars_system:app
```

### 3. 단일 추론 테스트

```bash
# 인텐트 분류
python src/intent_inference.py \
    --model_path models/intent_classifier \
    --text "짜장면 가격 알려주세요"

# 출력 예시:
# 예측 인텐트: 1 - 특정 상품 및 가격 안내
```

---

## 📊 모델 정보

### 인텐트 분류 모델
- **기본 모델**: KLUE/RoBERTa-Large (350M 파라미터)
- **클래스 수**: 48
- **학습 데이터**: `data/user_intent_v4.csv`
- **입력 형식**: 한국어 자연어 문장
- **출력**: 0-47 인텐트 ID

### NER 모델
- **기본 모델**: KLUE/RoBERTa-Large
- **엔티티**: 6개 (O, B-MENU, I-MENU, B-PAYMENT, I-PAYMENT, B-DAY)
- **학습 데이터**: CoNLL 형식 (token\tlabel)
- **출력**: BIO 태그 시퀀스

### 하이브리드 검색
- **BM25**: 40% 가중치 (키워드 기반)
- **SBERT**: 60% 가중치 (의미 기반, KoE5 모델)
- **총 점수**: `final_score = 0.4 * bm25 + 0.6 * sbert`

---

## 🐛 최근 수정 사항

### 2024-XX-XX: 인텐트 표준화 완료
- ✅ `intent_evaluation.py`: ID 33 중복 제거 (올바른 순서: 0-47)
- ✅ `response_templates.py`: 22개 → 48개 인텐트로 확장
- ✅ `intent_mapping.py`: 24개 → 48개 인텐트로 업데이트
- ✅ 모든 파일이 동일한 48개 인텐트 사용

### 이전 수정
- 하이브리드 검색 구현 (BM25 40% + SBERT 60%)
- Docker 배포 수정 (포트 9080, 모델 경로 수정)
- 파일명 리팩토링 (날짜 프리픽스 제거)

---

## 📝 TODO

- [ ] response_templates.py에 나머지 26개 인텐트 템플릿 구현
- [ ] 슬롯 필링 로직 강화 (NER 결과 활용)
- [ ] 대화 히스토리 관리 (Redis)
- [ ] A/B 테스트 프레임워크
- [ ] 멀티턴 대화 지원

---

## 🔗 관련 문서

- [API_USAGE.md](../API_USAGE.md) - API 엔드포인트 상세 사용법
- [README.md](../README.md) - 프로젝트 전체 개요
- [DATA_GUIDE.md](../DATA_GUIDE.md) - 데이터 형식 및 스키마

---

**작성일**: 2024-XX-XX  
**작성자**: GitHub Copilot  
**버전**: 1.0.0
