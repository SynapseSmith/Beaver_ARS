# Beaver ARS API 사용 가이드

## 📡 기본 정보
- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/json`
- **배포 방법**: Docker Compose

---

## 🔍 API 엔드포인트

### 1. Health Check
시스템 상태 확인

```bash
curl http://localhost:5000/health
```

**응답:**
```json
{
  "status": "healthy",
  "service": "beaver-ars"
}
```

---

### 2. Intent 분류 (`/predict`)
사용자 입력 텍스트의 의도(Intent)를 분류합니다.

**요청:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "짜장면 얼마예요?"
  }'
```

**응답:**
```json
{
  "text": "짜장면 얼마예요?",
  "intent_id": 1,
  "intent_name": "특정 상품 및 가격 안내"
}
```

**테스트 예시:**
```bash
# 메뉴 가격 문의
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "짬뽕 얼마예요?"}'

# 영업시간 문의
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "오늘 몇시까지 해요?"}'

# 결제 방법 문의
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "카드 결제 돼요?"}'
```

---

### 3. 개체명 인식 (`/ner`)
텍스트에서 메뉴, 결제수단, 요일 등의 엔티티를 추출합니다.

**요청:**
```bash
curl -X POST http://localhost:5000/ner \
  -H "Content-Type: application/json" \
  -d '{
    "text": "짜장면 2개랑 짬뽕 배달해주세요. 카드로 결제할게요"
  }'
```

**응답:**
```json
{
  "text": "짜장면 2개랑 짬뽕 배달해주세요. 카드로 결제할게요",
  "entities": [
    {
      "label": "MENU",
      "value": "짜장면"
    },
    {
      "label": "MENU",
      "value": "짬뽕"
    },
    {
      "label": "PAYMENT",
      "value": "카드로"
    }
  ]
}
```

**인식 가능한 엔티티:**
- `MENU`: 메뉴 이름
- `PAYMENT`: 결제 수단
- `DAY`: 요일

**테스트 예시:**
```bash
# 복잡한 주문
curl -X POST http://localhost:5000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "월요일에 탕수육이랑 짬뽕 주문하려고요. 네이버페이로 결제할게요"}'
```

---

### 4. 간단한 채팅 (`/chat`)
Intent 분류 + 응답 생성을 한 번에 처리하는 간소화된 API

**요청:**
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "짜장면 가격 알려줘"
  }'
```

**응답:**
```json
{
  "text": "짜장면 가격 알려줘",
  "intent_id": 1,
  "intent_name": "특정 상품 및 가격 안내",
  "response": "짜장의 가격은 6500원이에요. 더 자세한 내용은 문자로 발송된 메뉴판을 참고해주세요!",
  "processing_time": "0.1807 seconds"
}
```

**특징:**
- 하이브리드 검색 (BM25 40% + SBERT 60%) 적용
- NER 자동 처리
- 즉각적인 응답 생성

**테스트 예시:**
```bash
# 메뉴 가격
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "탕수육 얼마예요?"}'

# 메뉴 카테고리
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "어떤 메뉴가 있어요?"}'

# 영업 시간
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "오늘 몇시까지 영업하나요?"}'
```

---

### 5. 주문 처리 (`/order`)
전체 주문 처리 로직 (원본 API 포맷)

**요청:**
```bash
curl -X POST http://localhost:5000/order \
  -H "Content-Type: application/json" \
  -d '{
    "header": {
      "interfaceID": "AI-SDC-CAT-001"
    },
    "body": {
      "text": "짜장면 가격 알려줘"
    }
  }'
```

**응답:**
```json
{
  "response": "짜장의 가격은 6500원이에요. 더 자세한 내용은 문자로 발송된 메뉴판을 참고해주세요!",
  "processing_time": "0.9738 seconds"
}
```

**특징:**
- 완전한 주문 처리 워크플로우
- Excel 데이터베이스 쿼리
- 템플릿 기반 응답 생성

---

### 6. Intent 목록 조회 (`/intents`)
시스템에서 인식 가능한 모든 Intent 목록 조회

**요청:**
```bash
curl http://localhost:5000/intents
```

**응답:**
```json
{
  "total": 48,
  "intents": [
    {"id": 0, "name": "메뉴 카테고리 안내"},
    {"id": 1, "name": "특정 상품 및 가격 안내"},
    {"id": 2, "name": "상품에 대한 상세 및 추가 안내"},
    ...
  ]
}
```

**활용:**
- 시스템 capability 확인
- Intent ID와 이름 매핑 정보 획득

---

## 🧪 테스트 시나리오

### 시나리오 1: 메뉴 가격 문의
```bash
# 1. Intent 확인
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "짬뽕 얼마예요?"}'

# 2. 전체 응답 받기
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "짬뽕 얼마예요?"}'
```

### 시나리오 2: 복잡한 주문
```bash
# 1. NER로 엔티티 추출
curl -X POST http://localhost:5000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "월요일에 짜장면 2개 카드로 결제할게요"}'

# 2. 전체 주문 처리
curl -X POST http://localhost:5000/order \
  -H "Content-Type: application/json" \
  -d '{"header":{"interfaceID":"AI-SDC-CAT-001"},"body":{"text":"월요일에 짜장면 2개 카드로 결제할게요"}}'
```

### 시나리오 3: 하이브리드 검색 확인
```bash
# 유사 메뉴명으로 테스트
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "짜장 가격"}'  # "짜장면" 매칭

# 로그에서 BM25/SBERT 점수 확인
docker logs beaver-ars-app 2>&1 | grep "Hybrid Search"
```

---

## 📊 하이브리드 검색

### BM25 + SBERT 조합
- **BM25**: 키워드 기반 lexical 매칭 (가중치 40%)
- **SBERT**: 의미 기반 semantic 유사도 (가중치 60%)
- **최종 점수**: `0.4 × BM25 + 0.6 × SBERT`

### 로그 예시
```
Hybrid Search - Input: 짜장면, Match: 짜장, 
BM25: 0.964, SBERT: 0.809, Hybrid: 0.871
```

---

## 🐛 디버깅

### 로그 확인
```bash
# 실시간 로그
docker-compose logs -f app

# 최근 100줄
docker-compose logs app --tail=100

# 하이브리드 검색 점수만
docker-compose logs app | grep "Hybrid Search"

# Intent 분류 결과만
docker-compose logs app | grep "Intent ID"
```

### 컨테이너 상태
```bash
# 모든 서비스 상태
docker-compose ps

# 헬스체크
curl http://localhost:5000/health

# 메트릭
curl http://localhost:5000/metrics
```

---

## 🔧 운영 명령어

### 재시작
```bash
docker-compose restart app
```

### 재빌드
```bash
docker-compose build app
docker-compose stop app
docker-compose rm -f app
docker-compose up -d app
```

### 중지
```bash
docker-compose down
```

---

## 📝 Intent 목록 (48개)

<details>
<summary>전체 Intent 목록 보기</summary>

| ID | Intent 이름 |
|----|------------|
| 0 | 메뉴 카테고리 안내 |
| 1 | 특정 상품 및 가격 안내 |
| 2 | 상품에 대한 상세 및 추가 안내 |
| 3 | (뱃지) 인기메뉴 |
| 4 | (뱃지) 추천메뉴 |
| 5 | (뱃지) 대표메뉴 |
| 6 | (뱃지) 할인, 이벤트 |
| 7 | (뱃지) 신상 메뉴 |
| 8 | (뱃지) 한정 메뉴 |
| 9 | (뱃지) 매운맛 |
| 10 | (뱃지) 플러스 메뉴 |
| 11 | 주문 채널 |
| 12 | 주문 방법 |
| 13 | 결제 수단 |
| 14 | 요일별 운영 시간 |
| 15 | 매장 운영 방식 |
| 16 | 매장 위치 및 주차 |
| 17 | 배달 지역 |
| 18 | 배달 가능 여부 |
| 19 | 배달 소요 시간 |
| 20 | 배달비 |
| 21 | 최소 주문 금액 |
| 22 | 포장 가능 여부 |
| 23 | 픽업 가능 여부 |
| 24 | 배달 앱 문의 |
| 25 | 매장 주문 방법 |
| 26 | 포장 할인 |
| 27 | 주문 접수 확인 |
| 28 | 주문 취소 요청 |
| 29 | 주문 변경 요청 |
| 30 | 배달 지연 문의 |
| 31 | 배달 위치 변경 |
| 32 | 재배달 문의 |
| 33 | 영수증 및 현금 영수증 |
| 34 | 기타 문의 |
| 35 | 포인트 적립 문의 |
| 36 | 포인트 조회 문의 |
| 37 | 포인트 사용 문의 |
| 38 | 스탬프 적립 문의 |
| 39 | 쿠폰 사용 문의 |
| 40 | 기타 멤버십 문의 |
| 41 | 이벤트 안내 |
| 42 | 이벤트 관련 문의 |
| 43 | 기타 이벤트 문의 |
| 44 | 확인 후 답변 |
| 45 | 종료 인사 |
| 46 | 인삿말 |
| 47 | 알 수 없는 질문 |

</details>

---

## 💡 팁

1. **빠른 테스트**: `/chat` API 사용
2. **디버깅**: `/predict`와 `/ner`로 단계별 확인
3. **프로덕션**: `/order` API 사용 (표준 인터페이스)
4. **성능 확인**: `processing_time` 필드 참조
5. **하이브리드 검색**: 로그에서 BM25/SBERT 점수 모니터링
