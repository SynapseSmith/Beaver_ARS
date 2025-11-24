# 시스템 아키텍처 상세 문서

## 목차
1. [전체 시스템 구성](#전체-시스템-구성)
2. [Intent Classification 모듈](#intent-classification-모듈)
3. [NER 모듈](#ner-모듈)
4. [하이브리드 검색 엔진](#하이브리드-검색-엔진)
5. [응답 생성 시스템](#응답-생성-시스템)
6. [데이터 흐름](#데이터-흐름)
7. [성능 최적화](#성능-최적화)

---

## 전체 시스템 구성

### 시스템 개요
Beaver ARS는 마이크로서비스 아키텍처를 기반으로 한 모듈형 설계를 채택했습니다. 각 모듈은 독립적으로 개발, 테스트, 배포가 가능하며, 필요에 따라 확장할 수 있습니다.

```
┌───────────────────────────────────────────────────────────────┐
│                        Web Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐           │
│  │   Flask     │  │  WebSocket  │  │  Static      │           │
│  │   Server    │  │  Handler    │  │  Resources   │           │
│  └─────────────┘  └─────────────┘  └──────────────┘           │
└───────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┼───────────────────────────────────┐
│                    NLP Processing Layer                       │
│  ┌─────────────────────┐ │ ┌─────────────────────┐            │
│  │  Intent Classifier  │ │ │    NER Model        │            │
│  │  (RoBERTa-Large)    │ │ │  (RoBERTa-Large)    │            │
│  │  - 48 Classes       │ │ │  - 6 Entity Types   │            │
│  │  - Softmax Output   │ │ │  - Token Tagging    │            │
│  └─────────────────────┘ │ └─────────────────────┘            │
└───────────────────────────┼───────────────────────────────────┘
                            │
┌───────────────────────────┼───────────────────────────────────┐
│                    Search & Retrieval Layer                   │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐       │
│  │  BM25 Search  │  │  Semantic     │  │  Score       │       │
│  │  (Lexical)    │  │  Search       │  │  Combiner    │       │
│  │  - TF-IDF     │  │  (S-BERT)     │  │  (0.3 + 0.7) │       │
│  └───────────────┘  └───────────────┘  └──────────────┘       │
└───────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┼───────────────────────────────────┐
│                       Data Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐           │
│  │   SQL DB    │  │  Template   │  │  Cache       │           │
│  │   (MySQL)   │  │  Engine     │  │  (Redis)     │           │
│  └─────────────┘  └─────────────┘  └──────────────┘           │
└───────────────────────────────────────────────────────────────┘
```

### 주요 컴포넌트

#### 1. Web Server (Flask)
- **역할**: HTTP 요청 처리, 세션 관리, 정적 파일 제공
- **포트**: 5007 (웹 UI), 1117 (API)
- **특징**:
  - CORS 지원
  - JSON 기반 통신
  - 로깅 및 모니터링

#### 2. Intent Classifier
- **모델**: KLUE/RoBERTa-Large
- **입력**: 텍스트 문장 (최대 512 토큰)
- **출력**: 48차원 확률 벡터
- **처리 시간**: ~18ms (GPU), ~45ms (CPU)

#### 3. NER Model
- **모델**: KLUE/RoBERTa-Large (Token Classification)
- **태깅 방식**: IO Tagging
- **엔티티**: MENU, PAYMENT, DAY
- **처리 시간**: ~20ms (GPU), ~50ms (CPU)

#### 4. Search Engine
- **BM25**: 키워드 매칭 (k1=1.5, b=0.75)
- **Sentence-BERT**: 의미 벡터 유사도 (코사인 유사도)
- **결합**: Weighted sum (0.3 × BM25 + 0.7 × Semantic)

#### 5. Database
- **RDBMS**: MySQL 8.0+
- **스키마**: 메뉴, 가격, 카테고리, 이벤트 정보
- **인덱싱**: 메뉴명, 카테고리에 B-tree 인덱스

---

## Intent Classification 모듈

### 모델 아키텍처

```
Input Text
    │
    ▼
Tokenizer (WordPiece)
    │
    ▼
Token IDs + Attention Mask
    │
    ▼
RoBERTa Encoder (24 layers)
    │
    ├─→ [CLS] Token Representation
    │
    ▼
Linear Layer (768 → 48)
    │
    ▼
Softmax Activation
    │
    ▼
Intent Probabilities (48 classes)
```

### 학습 과정

#### 데이터셋 구성
- **총 샘플 수**: 3,524개
- **Train/Test 비율**: 80:20
- **클래스 분포**: 불균형 처리 (Class Weighting)

#### 하이퍼파라미터
```python
{
    "model": "klue/roberta-large",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 20,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_length": 128
}
```

#### 학습 곡선
- **Epoch 1-5**: Loss 급격히 감소 (2.5 → 0.8)
- **Epoch 6-15**: Loss 완만히 감소 (0.8 → 0.3)
- **Epoch 16-20**: Fine-tuning (0.3 → 0.2)

### 추론 프로세스

1. **전처리**
   ```python
   # 텍스트 정규화
   text = remove_special_chars(text)
   text = normalize_spaces(text)
   ```

2. **토큰화**
   ```python
   inputs = tokenizer(
       text,
       max_length=128,
       padding='max_length',
       truncation=True,
       return_tensors='pt'
   )
   ```

3. **모델 추론**
   ```python
   with torch.no_grad():
       outputs = model(**inputs)
       logits = outputs.logits
       probs = F.softmax(logits, dim=-1)
   ```

4. **후처리**
   ```python
   predicted_class = torch.argmax(probs, dim=-1)
   confidence = probs[0][predicted_class].item()
   ```

### 성능 평가

#### Confusion Matrix 분석
- **높은 정확도 클래스**: 인사(98%), 감사(97%), 주소 안내(96%)
- **혼동되는 클래스 쌍**:
  - 특정 메뉴 안내 ↔ 메뉴 카테고리 안내 (8% 혼동)
  - 영업시간 ↔ 특정 요일 영업시간 (5% 혼동)

#### 개선 방법
- 혼동되는 클래스 간 데이터 증강
- 하드 네거티브 마이닝
- Focal Loss 적용 고려

---

## NER 모듈

### Token Classification 방식

#### IO Tagging Scheme
```
Input:   떡볶이 가격이 얼마예요?
Tokens:  떡볶이 / 가격 / 이 / 얼마 / 예요 / ?
Tags:    B-MENU / O    / O / O    / O   / O
```

#### 모델 출력
각 토큰마다 6차원 확률 벡터 생성:
```python
{
    "떡볶이": [0.02, 0.95, 0.01, 0.00, 0.01, 0.01],  # B-MENU (95%)
    "가격이": [0.90, 0.02, 0.03, 0.01, 0.02, 0.02],  # O (90%)
    ...
}
```

### 학습 데이터 형식

#### CoNLL 형식
```conll
떡볶이	B-MENU
와	O
김밥	B-MENU
주세요	O

카드	B-PAYMENT
결제	I-PAYMENT
되나요	O
```

#### 데이터 통계
- **총 문장 수**: 1,850개
- **총 토큰 수**: 15,420개
- **엔티티 분포**:
  - MENU: 2,340개 (60%)
  - PAYMENT: 890개 (23%)
  - DAY: 670개 (17%)

### Entity Extraction 알고리즘

```python
def extract_entities(tokens, tags):
    entities = {}
    current_entity = []
    current_type = None
    
    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            # 이전 엔티티 저장
            if current_entity:
                entity_text = ''.join(current_entity)
                entities[current_type] = entity_text
            
            # 새 엔티티 시작
            current_type = tag.split('-')[1]
            current_entity = [token]
        
        elif tag.startswith('I-'):
            current_entity.append(token)
        
        elif tag == 'O':
            # 엔티티 종료
            if current_entity:
                entity_text = ''.join(current_entity)
                entities[current_type] = entity_text
                current_entity = []
                current_type = None
    
    return entities
```

### 성능 지표

#### Entity-level F1-Score
```
              precision    recall  f1-score   support

      MENU       0.942     0.928     0.935       234
   PAYMENT       0.915     0.893     0.904        89
       DAY       0.961     0.957     0.959        67

  macro avg       0.939     0.926     0.933       390
```

---

## 하이브리드 검색 엔진

### BM25 (Okapi BM25)

#### 알고리즘
```
BM25(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))

where:
- D: Document (응답 후보)
- Q: Query (사용자 입력)
- f(qi, D): qi가 D에 등장하는 빈도
- |D|: 문서 길이
- avgdl: 평균 문서 길이
- k1: 1.5 (term frequency saturation)
- b: 0.75 (length normalization)
```

#### 구현
```python
from rank_bm25 import BM25Okapi

# 문서 코퍼스 토큰화
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 쿼리 검색
query_tokens = query.split()
bm25_scores = bm25.get_scores(query_tokens)
```

### Sentence-BERT Semantic Search

#### 모델
- **사전학습 모델**: `jhgan/ko-sbert-nli`
- **임베딩 차원**: 768
- **유사도 메트릭**: Cosine Similarity

#### 프로세스
```python
from sentence_transformers import SentenceTransformer, util

# 모델 로드
model = SentenceTransformer('jhgan/ko-sbert-nli')

# 문서 임베딩 (캐싱)
doc_embeddings = model.encode(corpus, convert_to_tensor=True)

# 쿼리 임베딩
query_embedding = model.encode(query, convert_to_tensor=True)

# 코사인 유사도 계산
cosine_scores = util.cos_sim(query_embedding, doc_embeddings)
```

### 하이브리드 점수 결합

#### 점수 정규화
```python
def normalize_scores(scores):
    """Min-Max Normalization"""
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score + 1e-10)
```

#### 가중 합산
```python
# 하이퍼파라미터
alpha = 0.3  # BM25 가중치
beta = 0.7   # Semantic 가중치

# 점수 정규화
bm25_norm = normalize_scores(bm25_scores)
semantic_norm = normalize_scores(semantic_scores)

# 최종 점수
final_scores = alpha * bm25_norm + beta * semantic_norm

# Top-K 선택
top_k_idx = np.argsort(final_scores)[-k:][::-1]
```

#### 가중치 실험 결과
| α (BM25) | β (Semantic) | MRR@10 | NDCG@10 |
|----------|--------------|--------|---------|
| 0.5      | 0.5          | 0.783  | 0.812   |
| 0.3      | 0.7          | **0.821** | **0.845** |
| 0.2      | 0.8          | 0.809  | 0.831   |
| 0.0      | 1.0          | 0.791  | 0.818   |

---

## 응답 생성 시스템

### Template 기반 생성

#### Template 구조
```python
response_templates = {
    1: [  # 특정 메뉴 안내
        "{STD_MENU_NM}의 가격은 {PRICE}원이에요. 더 자세한 내용은 문자로 발송된 메뉴판을 참고해주세요!",
        "{STD_MENU_NM}은 {PRICE}원에 판매 중입니다.",
    ],
    3: [  # 인기 메뉴
        "요즘 {POPULAR_MENU} 메뉴가 가장 {INDI_TYPE_NM5} 있는 메뉴입니다.",
    ],
    ...
}
```

### Slot Filling 과정

#### 1. Entity Extraction
```python
entities = ner_model.extract(user_input)
# entities = {"MENU": ["떡볶이"]}
```

#### 2. Database Query
```python
def query_menu_info(menu_name):
    query = """
    SELECT m.STD_MENU_NM, m.PRICE, c.STD_CATEGORY_NM
    FROM menu m
    JOIN category c ON m.category_id = c.id
    WHERE m.STD_MENU_NM LIKE %s
    """
    cursor.execute(query, (f"%{menu_name}%",))
    return cursor.fetchone()
```

#### 3. Template Filling
```python
def fill_template(template, slot_values):
    """슬롯 값으로 템플릿 채우기"""
    response = template
    for slot, value in slot_values.items():
        placeholder = "{" + slot + "}"
        response = response.replace(placeholder, str(value))
    return response
```

#### 4. Fallback 처리
```python
if confidence < 0.6:
    return fallback_response()
elif missing_slots:
    return ask_clarification(missing_slots)
else:
    return filled_template
```

### Response Ranking

#### 다중 템플릿 선택
```python
def select_best_template(intent_id, context, templates):
    """컨텍스트에 가장 적합한 템플릿 선택"""
    scores = []
    for template in templates[intent_id]:
        # 필요한 슬롯이 모두 채워질 수 있는지 확인
        required_slots = extract_slots(template)
        available_slots = get_available_slots(context)
        
        if all(slot in available_slots for slot in required_slots):
            scores.append(1.0)
        else:
            scores.append(0.0)
    
    best_idx = np.argmax(scores)
    return templates[intent_id][best_idx]
```

---

## 데이터 흐름

### 전체 요청 처리 프로세스

```
1. User Input
   "떡볶이 가격이 얼마예요?"
   │
   ▼
2. Preprocessing
   - 정규화: "떡볶이 가격이 얼마예요"
   - 토큰화: ["떡볶이", "가격", "이", "얼마", "예요"]
   │
   ▼
3. Intent Classification (18ms)
   - Input: "떡볶이 가격이 얼마예요"
   - Output: intent_id=1 (특정 메뉴 안내), confidence=0.98
   │
   ▼
4. NER Tagging (20ms)
   - Tokens: ["떡볶이", "가격", "이", "얼마", "예요"]
   - Tags: ["B-MENU", "O", "O", "O", "O"]
   - Entities: {"MENU": "떡볶이"}
   │
   ▼
5. Template Selection
   - Intent: 특정 메뉴 안내
   - Template: "{STD_MENU_NM}의 가격은 {PRICE}원이에요."
   │
   ▼
6. Hybrid Search (45ms)
   - BM25 Search: "떡볶이" → top_k candidates
   - Semantic Search: embedding similarity
   - Combined Score: 0.3*BM25 + 0.7*Semantic
   │
   ▼
7. Database Query (12ms)
   - Query: SELECT * FROM menu WHERE STD_MENU_NM='떡볶이'
   - Result: {name: "떡볶이", price: 5000, ...}
   │
   ▼
8. Slot Filling
   - Slots: {STD_MENU_NM: "떡볶이", PRICE: 5000}
   - Response: "떡볶이의 가격은 5,000원이에요."
   │
   ▼
9. Post-processing
   - 숫자 포맷팅: 5000 → "5,000"
   - 어미 처리
   │
   ▼
10. Response Output
    "떡볶이의 가격은 5,000원이에요. 더 자세한 내용은 문자로 발송된 메뉴판을 참고해주세요!"
```

### 평균 처리 시간 분해
```
Component               Time (ms)    Percentage
─────────────────────────────────────────────
Preprocessing                 5ms         0.6%
Intent Classification        18ms         2.1%
NER Tagging                  20ms         2.4%
Template Selection            2ms         0.2%
Hybrid Search                45ms         5.3%
Database Query               12ms         1.4%
Slot Filling                  3ms         0.4%
Post-processing               5ms         0.6%
Network I/O                 740ms        87.0%
─────────────────────────────────────────────
Total                       850ms       100.0%
```

---

## 성능 최적화

### 1. 모델 최적화

#### Quantization (INT8)
```python
from torch.quantization import quantize_dynamic

# Dynamic Quantization
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 추론 속도: 18ms → 12ms (33% 개선)
# 정확도 하락: 95.7% → 95.3% (0.4% 하락)
```

#### ONNX Runtime
```python
import onnxruntime as ort

# 모델 변환
torch.onnx.export(model, dummy_input, "model.onnx")

# ONNX Runtime 추론
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {input_name: input_data})

# 추론 속도: 18ms → 10ms (44% 개선)
```

### 2. 캐싱 전략

#### Response Cache (Redis)
```python
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_response(query):
    # 캐시 키 생성
    cache_key = f"response:{hash(query)}"
    
    # 캐시 조회
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 캐시 미스 - 새로 생성
    response = generate_response(query)
    cache.setex(cache_key, 3600, json.dumps(response))  # 1시간 TTL
    return response
```

#### Embedding Cache
```python
# 문서 임베딩 사전 계산 및 저장
doc_embeddings = model.encode(corpus)
np.save('embeddings.npy', doc_embeddings)

# 로딩 시간: 5초 → 0.1초 (50배 개선)
```

### 3. 병렬 처리

#### Batch Inference
```python
def batch_predict(texts, batch_size=32):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_outputs = model(batch)
        predictions.extend(batch_outputs)
    return predictions

# 처리량: 100 req/s → 350 req/s (3.5배 개선)
```

#### Multi-threading
```python
from concurrent.futures import ThreadPoolExecutor

def process_request(request):
    # Intent & NER 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        intent_future = executor.submit(intent_classifier, request)
        ner_future = executor.submit(ner_model, request)
        
        intent = intent_future.result()
        entities = ner_future.result()
    
    return intent, entities

# 처리 시간: 38ms → 22ms (42% 개선)
```

### 4. Database 최적화

#### Index 생성
```sql
-- 메뉴명 인덱스
CREATE INDEX idx_menu_name ON menu(STD_MENU_NM);

-- 카테고리 인덱스
CREATE INDEX idx_category ON menu(category_id);

-- 복합 인덱스
CREATE INDEX idx_menu_category ON menu(STD_MENU_NM, category_id);

-- 쿼리 속도: 50ms → 12ms (76% 개선)
```

#### Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'mysql://user:password@localhost/beaver_ars',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# 연결 시간: 30ms → 5ms (83% 개선)
```

### 성능 개선 요약

| 최적화 기법 | Before | After | 개선율 |
|------------|--------|-------|--------|
| Model Quantization | 18ms | 12ms | 33% |
| ONNX Runtime | 18ms | 10ms | 44% |
| Response Cache | 850ms | 50ms | 94% (캐시 적중 시) |
| Batch Inference | 100 req/s | 350 req/s | 250% |
| DB Indexing | 50ms | 12ms | 76% |
| Connection Pool | 30ms | 5ms | 83% |

---

## 확장성 및 유지보수

### Horizontal Scaling
```
Load Balancer (Nginx)
        │
    ┌───┴───┬───────┬───────┐
    │       │       │       │
  API-1  API-2  API-3  API-N
    │       │       │       │
    └───┬───┴───┬───┴───┬───┘
        │       │       │
    Shared Redis Cache
        │
    Database Cluster
```

### Monitoring & Logging
```python
from prometheus_client import Counter, Histogram

# 메트릭 수집
request_count = Counter('requests_total', 'Total requests')
response_time = Histogram('response_time_seconds', 'Response time')

@app.route('/order', methods=['POST'])
@response_time.time()
def order():
    request_count.inc()
    # ... 처리 로직
```

---

## 보안 고려사항

### 1. Input Validation
```python
def validate_input(text):
    # 길이 제한
    if len(text) > 500:
        raise ValueError("Input too long")
    
    # SQL Injection 방지
    if any(keyword in text.lower() for keyword in ['drop', 'delete', 'update']):
        raise ValueError("Invalid input")
```

### 2. Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour"]
)
```

### 3. API Authentication
```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != VALID_API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

---

**문서 버전**: 1.0  
**최종 수정일**: 2025-01-02  
**작성자**: Beaver ARS Team
