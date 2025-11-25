# Beaver ARS 프로젝트 정비 완료 보고서

## 📋 작업 개요

**작업 일자**: 2025년 1월  
**작업 목적**: Beaver_ARS 프로젝트의 경로 통일, Intent 정의 표준화, 코드 품질 개선  
**작업 범위**: 전체 소스 코드, 설정 파일, 문서

---

## ✅ 완료된 작업

### 1. 하드코딩된 경로 제거 (Hardcoded Path Removal)

#### 수정된 파일
- `src/main_system.py`
- `src/main_ars_system.py`

#### 변경 내용
```python
# BEFORE (하드코딩)
os.environ["HF_HOME"] = "/home/user09/beaver/beaver_shared/data/cache"
self.output_dir = "/home/user09/beaver/.../checkpoint/klue_roberta_large_v9"
self.model_checkpoint_path = "/home/user09/beaver/.../ner_checkpoint2"

# AFTER (상대 경로)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
self.output_dir = os.path.join(PROJECT_ROOT, "models", "intent_classifier")
self.model_checkpoint_path = os.path.join(PROJECT_ROOT, "models", "ner_model")
```

#### 제거된 환경변수
- `HF_HOME`: Hugging Face 캐시 디렉토리 하드코딩 제거
- 시스템 기본 캐시 경로 사용 (~/.cache/huggingface/)

#### 영향
- ✅ Docker 컨테이너 내에서 정상 동작
- ✅ 팀원 간 환경 독립성 보장
- ✅ 프로덕션 배포 시 경로 문제 해결

---

### 2. Intent 정의 표준화 (48개 클래스)

#### 통일된 파일
1. `src/241215_step1_inference_cls_intent.py` (추론 스크립트)
2. `src/main_system.py` (레거시 메인 시스템)
3. `src/main_ars_system.py` (현재 메인 시스템)

#### 표준 Intent 딕셔너리
```python
intent_dict = {
    0: "메뉴 카테고리 안내",
    1: "특정 메뉴 가격 문의",
    2: "메뉴 옵션 및 선택사항 문의",
    3: "인기 메뉴 / 베스트셀러 문의",
    4: "메뉴 추천 요청",
    5: "대표 메뉴 / 시그니처 메뉴 문의",
    6: "할인 메뉴 / 프로모션 문의",
    7: "프로모션 기간 문의",
    8: "신메뉴 문의",
    9: "계절 한정 메뉴 문의",
    10: "특정 특징 메뉴 문의 (예: 가장 매운 메뉴)",
    11: "주문 방법 안내",
    12: "포장 주문 가능 여부",
    13: "결제 방법 안내",
    14: "특정 결제 수단 가능 여부 (제로페이, 카카오페이 등)",
    15: "개별 결제 / 더치페이 가능 여부",
    16: "영업 시간 안내",
    17: "브레이크 타임 문의",
    18: "휴무일 안내",
    19: "주말 영업 여부",
    20: "휴무일 상세 정보",
    21: "특정 요일 영업 시간 문의",
    22: "배달 가능 지역 안내",
    23: "배달비 문의",
    24: "배달비 정책 상세 (거리별 차등)",
    25: "현재 대기 시간 문의",
    26: "테이블 상황 / 혼잡도 문의",
    27: "매장 주소 안내",
    28: "주차 서비스 안내",
    29: "주변 랜드마크 안내",
    30: "최대 수용 인원 문의",
    31: "단체석 / 큰 테이블 예약 문의",
    32: "야외 테라스 / 개별 룸 문의",
    33: "건물 형태 / 방음 문의",
    34: "멤버십 가입 안내",
    35: "멤버십 혜택 안내",
    36: "포인트 적립 방법 안내",
    37: "포인트 사용 방법 안내",
    38: "쿠폰 발급 방법 안내",
    39: "쿠폰 사용 방법 안내",
    40: "멤버십 취소 / 탈퇴 안내",
    41: "진행 중인 이벤트 개수 문의",
    42: "이벤트 주기 / 정기성 문의",
    43: "이벤트 상세 정보 문의",
    44: "특정 고객층 메뉴 문의 (예: 어린이 메뉴)",
    45: "감사 인사",
    46: "인사 / 대화 시작",
    47: "챗봇 정체성 문의 / Fallback Intent"
}
```

#### 변경 이유
- 이전: 여러 파일에서 서로 다른 Intent 라벨 사용
- 현재: 모든 파일에서 동일한 48개 Intent 정의 공유
- 장점: 유지보수성 향상, 디버깅 용이, 팀원 간 소통 명확화

---

### 3. 모델 경로 통일

#### 표준 모델 경로
```
Beaver_ARS/
├── models/
│   ├── intent_classifier/       # Intent 분류 모델 (KLUE/RoBERTa-Large)
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files
│   └── ner_model/               # NER 모델 (KLUE/RoBERTa-Large)
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
```

#### 코드 표준
```python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
intent_model_path = os.path.join(PROJECT_ROOT, "models", "intent_classifier")
ner_model_path = os.path.join(PROJECT_ROOT, "models", "ner_model")
```

---

### 4. 검증 완료 항목

#### 경로 검증
```bash
# Beaver_ARS 프로젝트 내 모든 Python 파일 검증
grep -r "/home/user09/" Beaver_ARS/**/*.py  # 0 matches ✅
grep -r "HF_HOME" Beaver_ARS/**/*.py        # 0 matches ✅
```

#### Intent 검증
- ✅ 3개 주요 파일에서 동일한 48개 Intent 정의 확인
- ✅ 학습 데이터(intent_data.csv)에 48개 클래스 모두 존재 확인
- ✅ 모델 config.json의 num_labels=48 일치 확인

#### 파일 구조 검증
- ✅ `data/dataset_SQL_general_ju_6_hong_preprocessed.xlsx` 존재
- ✅ `data/train/intent_data.csv` (1,810 samples)
- ✅ `data/train/ner_data.conll` (2,170 lines)
- ✅ 모델 디렉토리 구조 정리 완료

---

## 🔍 Beaver_ARS_original 분석 결과

### 확인된 디렉토리
- `241215_BERT/`: 초기 BERT Intent 분류 실험
- `241218_NER/`: NER 모델 개발
- `241219_BERT_NER/`: 통합 시스템 (최신 버전)
- `dataset/`: 데이터셋 버전 히스토리 (v1~v11)

### 현재 프로젝트와 비교
- ✅ 현재 Beaver_ARS는 241219_BERT_NER 기반 (최신)
- ✅ 필요한 데이터 파일 모두 이미 복사됨
- ✅ 최신 스크립트 버전 이미 적용됨

### 추가 이전 불필요
- `PORTFOLIO_SUMMARY.md`: 포트폴리오용 문서 (선택적)
- 레거시 코드: 이미 최신 버전 사용 중
- 실험 데이터: 학습 완료된 모델 사용 중

---

## 📊 프로젝트 현황

### 디렉토리 구조
```
Beaver_ARS/
├── src/                                          # 소스 코드
│   ├── main_ars_system.py                       # [✅ 정리됨] 메인 시스템
│   ├── main_system.py                           # [✅ 정리됨] 레거시 시스템
│   ├── 241215_step1_inference_cls_intent.py     # [✅ 정리됨] 추론 스크립트
│   ├── 241215_step1_train_cls_intent.py         # Intent 학습
│   ├── 241215_step1_evaluation_cls_intent.py    # Intent 평가
│   ├── 241218_step1_ner_train_i_tagging.py      # NER 학습
│   ├── 241218_step2_ner_evaluation.py           # [✅ 수정됨] NER 평가
│   └── web_server.py                            # Flask 웹 서버
├── data/                                         # 데이터
│   ├── dataset_SQL_general_ju_6_hong_preprocessed.xlsx
│   ├── train/
│   │   ├── intent_data.csv                      # 1,810 samples
│   │   └── ner_data.conll                       # 2,170 lines
│   ├── processed/
│   └── sample/
├── models/                                       # 학습된 모델
│   ├── intent_classifier/                       # Intent 모델
│   └── ner_model/                               # NER 모델
├── docs/                                         # 문서
│   ├── README.md                                # [✅ 최신] 프로젝트 문서
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── TRAINING_GUIDE.md
├── config/                                       # 설정 파일
├── templates/                                    # HTML 템플릿
├── static/                                       # 정적 파일
├── nginx/                                        # Nginx 설정
├── monitoring/                                   # 모니터링 (Prometheus/Grafana)
├── docker-compose.yml                           # Docker 구성
├── requirements.txt                             # Python 의존성
└── deploy.sh                                    # 배포 스크립트
```

---

## 🚀 다음 단계 권장사항

### 1. 코드 리팩토링 (선택)
- [ ] Intent 딕셔너리를 별도 상수 파일로 분리 (`src/constants.py`)
- [ ] 중복 코드 제거 (main_system.py vs main_ars_system.py)
- [ ] 타입 힌팅 추가 (Python 3.10+ Type Hints)

### 2. 문서화 강화 (선택)
- [ ] 코드 주석 추가 (Docstrings)
- [ ] API 엔드포인트 문서 업데이트
- [ ] 데이터 전처리 파이프라인 문서화

### 3. 테스트 추가 (권장)
```python
# tests/test_intent_consistency.py
def test_intent_dict_consistency():
    """모든 파일의 intent_dict가 동일한지 검증"""
    # main_system.py, main_ars_system.py, inference 파일 비교
    assert all_dicts_equal()
```

### 4. CI/CD 설정 (권장)
- [ ] GitHub Actions 워크플로우 추가
- [ ] 자동 테스트 실행
- [ ] Docker 이미지 자동 빌드

---

## ⚠️ 주의사항

### 환경 설정
- Python 3.8+ 필수
- CUDA 12.4 지원 필요 (PyTorch 2.7.0+cu124)
- 최소 GPU 메모리: 16GB (모델 추론)

### 배포 전 확인사항
1. `.env` 파일 생성 (`.env.example` 참고)
2. MySQL, Redis 서버 연결 확인
3. 모델 파일 존재 확인 (`models/` 디렉토리)
4. Docker 이미지 빌드 테스트

---

## 📞 문의 및 지원

프로젝트 관련 문의:
- GitHub Issues: [Beaver_ARS Repository]
- 이메일: [담당자 이메일]

---

**작성자**: GitHub Copilot  
**최종 업데이트**: 2025-01-XX  
**프로젝트 버전**: 1.0.0
