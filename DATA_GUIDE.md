# 📊 Beaver ARS 데이터셋 가이드

## 📁 데이터 구조

```
data/
├── train/                          # 실제 학습 데이터 (메인)
│   ├── intent_data.csv            # Intent 분류 데이터 (1,810개)
│   └── ner_data.conll             # NER 학습 데이터 (2,170줄)
│
├── sample/                         # 샘플 데이터 (테스트/데모용)
│   ├── intent_sample.csv          # Intent 샘플
│   └── ner_sample.conll           # NER 샘플
│
└── processed/                      # 전처리된 데이터 (학습 시 자동 생성)
    ├── intent_train.csv
    ├── intent_val.csv
    ├── intent_test.csv
    ├── ner_train.conll
    ├── ner_val.conll
    └── ner_test.conll
```

---

## 📊 데이터 통계

### Intent Classification Dataset

**파일**: `data/train/intent_data.csv`

| 항목 | 값 |
|------|-----|
| **총 샘플 수** | 1,810개 |
| **Intent 클래스 수** | 48개 |
| **형식** | CSV (user, intent_num) |
| **크기** | 83 KB |

**Intent 번호 예시**:
- `3`: 인기 메뉴 추천
- `5`: 메뉴 가격 문의
- `7`: 주문하기
- `15`: 영업시간 문의
- 등 48개 클래스

**샘플 데이터**:
```csv
user,intent_num
가장 많이 팔리는 메뉴는 뭔가요?,3
김치찌개 2개 주문할게요,7
영업시간이 언제인가요?,15
```

### NER Dataset

**파일**: `data/train/ner_data.conll`

| 항목 | 값 |
|------|-----|
| **총 라인 수** | 2,170줄 |
| **엔티티 타입** | 주로 MENU (음식명) |
| **형식** | CoNLL (BIO 태깅) |
| **크기** | 22 KB |

**엔티티 타입**:
- `B-MENU`: 메뉴명 시작
- `I-MENU`: 메뉴명 중간/끝
- `O`: 일반 토큰 (엔티티 아님)

**샘플 데이터**:
```
불고기    B-MENU
가격      O
은요?     O

김치찌개  B-MENU
값        O
은        O
얼마인가요? O
```

---

## 🎯 학습 데이터 사용법

### 1. Intent Classification 학습

```bash
# 전체 데이터로 학습
python src/241215_step1_train_cls_intent.py \
    --data_path data/train/intent_data.csv \
    --output_dir models/intent_classifier \
    --num_epochs 20 \
    --batch_size 16

# 예상 소요 시간:
# - GPU (RTX 3090): ~2시간
# - CPU: ~8-12시간
```

### 2. NER 학습

```bash
# 전체 데이터로 학습
python src/241218_step1_ner_train_i_tagging.py \
    --data_path data/train/ner_data.conll \
    --output_dir models/ner_model \
    --num_epochs 15 \
    --batch_size 16

# 예상 소요 시간:
# - GPU (RTX 3090): ~1.5시간
# - CPU: ~6-10시간
```

### 3. 빠른 테스트 (샘플 데이터)

```bash
# 샘플 데이터로 빠른 테스트 (3 epochs)
python train_quick.py --train_all --epochs 3 \
    --intent_data data/sample/intent_sample.csv \
    --ner_data data/sample/ner_sample.conll

# 소요 시간:
# - GPU: ~3-5분
# - CPU: ~15-30분
```

---

## 📝 데이터 형식 상세

### Intent CSV 형식

```csv
user,intent_num
{사용자 질문},{intent 번호(0-47)}
```

**예시**:
```csv
user,intent_num
가장 많이 팔리는 메뉴는 뭔가요?,3
김치찌개 2개 주문할게요,7
영업시간이 언제인가요?,15
배달 가능한가요?,18
```

### NER CoNLL 형식

```
{토큰}\t{태그}
{토큰}\t{태그}
(빈 줄로 문장 구분)
```

**예시**:
```
불고기    B-MENU
가격      O
은요?     O

김치찌개  B-MENU
2         B-QUANTITY
개        I-QUANTITY
주문      O
할게요    O
```

---

## 🔧 데이터 전처리

### 자동 Train/Val/Test 분할

```bash
# 데이터 전처리 및 분할 (70/15/15)
python scripts/data_preprocessing.py \
    --input data/train/intent_data.csv \
    --output data/processed/ \
    --split_ratio 0.7 0.15 0.15

# 결과:
# data/processed/intent_train.csv (1,267개)
# data/processed/intent_val.csv (271개)
# data/processed/intent_test.csv (272개)
```

### 데이터 검증

```bash
# Intent 데이터 검증
python scripts/data_preprocessing.py \
    --validate \
    --input data/train/intent_data.csv

# 확인 사항:
# ✓ CSV 형식 유효성
# ✓ Intent 번호 범위 (0-47)
# ✓ 중복 데이터 확인
# ✓ 클래스 불균형 확인
```

---

## 📊 데이터 분석

### Intent 분포 확인

```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('data/train/intent_data.csv')

# Intent 분포
print(df['intent_num'].value_counts().sort_index())

# 클래스 불균형 확인
print(f"최대: {df['intent_num'].value_counts().max()}")
print(f"최소: {df['intent_num'].value_counts().min()}")
```

### NER 엔티티 분포

```bash
# 엔티티 타입별 개수
grep "B-" data/train/ner_data.conll | cut -f2 | sort | uniq -c

# 예상 출력:
#  1850 B-MENU
#   120 B-QUANTITY
#    80 B-OPTION
```

---

## 🎯 데이터 증강 (선택사항)

### Back Translation

```python
# 한국어 → 영어 → 한국어
# 데이터 2배 증강 가능
python scripts/augment_data.py \
    --input data/train/intent_data.csv \
    --output data/train/intent_data_augmented.csv \
    --method back_translation
```

### Synonym Replacement

```python
# 유의어 치환
python scripts/augment_data.py \
    --input data/train/intent_data.csv \
    --output data/train/intent_data_augmented.csv \
    --method synonym
```

---

## 🔍 데이터 품질 확인

### Intent 데이터

✅ **확인 항목**:
- [ ] 1,800+ 샘플
- [ ] 48개 클래스 모두 커버
- [ ] 클래스당 최소 30개 샘플
- [ ] 중복 데이터 없음
- [ ] 라벨링 오류 없음

### NER 데이터

✅ **확인 항목**:
- [ ] 2,000+ 라인
- [ ] BIO 태깅 일관성
- [ ] 엔티티 경계 정확성
- [ ] 빈 줄로 문장 구분
- [ ] 탭으로 토큰-태그 구분

---

## 📚 추가 데이터 준비 (선택사항)

### 자신의 데이터 준비하기

#### Intent 데이터

1. **CSV 파일 생성**:
   ```csv
   user,intent_num
   {사용자 질문},{intent 번호}
   ```

2. **Intent 번호 매핑**:
   - `src/241215_step0_intent_mapping.py` 참조
   - 0-47 범위 내 번호 사용

3. **데이터 검증**:
   ```bash
   python scripts/data_preprocessing.py --validate --input your_data.csv
   ```

#### NER 데이터

1. **CoNLL 형식 생성**:
   ```
   토큰\t태그
   토큰\t태그
   
   토큰\t태그
   ```

2. **태그 규칙**:
   - `B-MENU`: 메뉴명 시작
   - `I-MENU`: 메뉴명 계속
   - `O`: 일반 토큰

3. **Annotation 도구**:
   - [doccano](https://github.com/doccano/doccano)
   - [Label Studio](https://labelstud.io/)

---

## 💡 팁

### 학습 속도 향상

```bash
# 작은 데이터셋으로 빠른 실험
head -200 data/train/intent_data.csv > data/sample/intent_quick.csv
python train_quick.py --intent_data data/sample/intent_quick.csv --epochs 3
```

### 데이터 품질 체크

```bash
# Intent 데이터 통계
python scripts/data_preprocessing.py --stats --input data/train/intent_data.csv

# 출력:
# Total samples: 1,810
# Unique intents: 48
# Avg samples per intent: 37.7
# Min samples: 25
# Max samples: 65
```

---

## 🔗 관련 파일

- **Intent 매핑**: `src/241215_step0_intent_mapping.py`
- **데이터 전처리**: `scripts/data_preprocessing.py`
- **학습 스크립트**: `train.sh`, `train_quick.py`
- **학습 가이드**: `TRAINING_WORKFLOW.md`

---

## 📞 문제 해결

### 데이터를 찾을 수 없음

```bash
# 데이터 확인
ls -lh data/train/

# 없으면 원본에서 복사
cp ../Beaver_ARS/241215_BERT/data/user_intent_v6.csv data/train/intent_data.csv
```

### 형식 오류

```bash
# CSV 인코딩 확인
file data/train/intent_data.csv

# UTF-8로 변환
iconv -f EUC-KR -t UTF-8 data.csv > data_utf8.csv
```

---

**📊 데이터가 준비되었습니다! 이제 학습을 시작하세요:**

```bash
# 전체 파이프라인
./train.sh

# 또는 빠른 테스트
python train_quick.py --train_all --epochs 3
```
