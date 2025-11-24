# API Reference - Beaver ARS

## 개요
Beaver ARS는 RESTful API를 제공하며, JSON 형식의 요청/응답을 사용합니다.

**Base URL**: `http://localhost:1117` (로컬 개발)  
**Content-Type**: `application/json; charset=utf-8`

---

## 인증
현재 버전에서는 API Key 기반 인증을 지원합니다.

```http
X-API-Key: your_api_key_here
```

---

## Endpoints

### 1. 주문/문의 처리 (Main API)

#### `POST /order`

고객의 텍스트 입력을 받아 Intent Classification, NER, 검색, 응답 생성을 수행하고 최종 응답을 반환합니다.

**Request**

```http
POST /order HTTP/1.1
Host: localhost:1117
Content-Type: application/json; charset=utf-8

{
  "header": {
    "interfaceID": "AI-SDC-CAT-001",
    "timestamp": "2025-01-02T10:30:00Z",
    "requestId": "req_123456"
  },
  "body": {
    "text": "떡볶이 가격이 얼마예요?",
    "userId": "user_001",
    "sessionId": "session_789"
  }
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `header.interfaceID` | string | Yes | 인터페이스 식별자 (고정값: "AI-SDC-CAT-001") |
| `header.timestamp` | string | No | 요청 시각 (ISO 8601 형식) |
| `header.requestId` | string | No | 요청 고유 ID (추적용) |
| `body.text` | string | Yes | 사용자 입력 텍스트 (최대 500자) |
| `body.userId` | string | No | 사용자 ID |
| `body.sessionId` | string | No | 세션 ID (대화 컨텍스트 관리) |

**Response**

```json
{
  "statusCode": 200,
  "header": {
    "responseId": "resp_123456",
    "timestamp": "2025-01-02T10:30:01Z",
    "processingTime": 850
  },
  "body": {
    "response": "떡볶이의 가격은 5,000원이에요. 더 자세한 내용은 문자로 발송된 메뉴판을 참고해주세요!",
    "intent": {
      "id": 1,
      "name": "특정 상품 및 가격 안내",
      "confidence": 0.98
    },
    "entities": {
      "MENU": ["떡볶이"]
    },
    "alternatives": [
      {
        "response": "떡볶이는 5,000원에 판매 중입니다.",
        "confidence": 0.85
      }
    ]
  }
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `statusCode` | integer | HTTP 상태 코드 |
| `header.responseId` | string | 응답 고유 ID |
| `header.timestamp` | string | 응답 시각 |
| `header.processingTime` | integer | 처리 시간 (ms) |
| `body.response` | string | 최종 생성된 응답 텍스트 |
| `body.intent.id` | integer | Intent 클래스 ID (0-47) |
| `body.intent.name` | string | Intent 클래스 이름 |
| `body.intent.confidence` | float | Intent 분류 신뢰도 (0-1) |
| `body.entities` | object | 추출된 엔티티 (키: 엔티티 타입, 값: 엔티티 리스트) |
| `body.alternatives` | array | 대안 응답 (선택적) |

**Status Codes**

| Code | Meaning |
|------|---------|
| 200 | 성공 |
| 400 | 잘못된 요청 (파라미터 오류) |
| 401 | 인증 실패 |
| 429 | Rate Limit 초과 |
| 500 | 서버 내부 오류 |
| 503 | 서비스 일시 중단 |

**Example cURL**

```bash
curl -X POST http://localhost:1117/order \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "header": {
      "interfaceID": "AI-SDC-CAT-001"
    },
    "body": {
      "text": "떡볶이 가격이 얼마예요?"
    }
  }'
```

**Example Python**

```python
import requests
import json

url = "http://localhost:1117/order"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your_api_key"
}
payload = {
    "header": {
        "interfaceID": "AI-SDC-CAT-001"
    },
    "body": {
        "text": "떡볶이 가격이 얼마예요?"
    }
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

---

### 2. 헬스 체크

#### `GET /health`

서버 상태를 확인합니다.

**Request**

```http
GET /health HTTP/1.1
Host: localhost:1117
```

**Response**

```json
{
  "status": "healthy",
  "timestamp": "2025-01-02T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "intentClassifier": "ready",
    "nerModel": "ready",
    "database": "connected",
    "cache": "connected"
  },
  "uptime": 3600,
  "requestsProcessed": 1523
}
```

---

### 3. Intent 목록 조회

#### `GET /intents`

지원하는 모든 Intent 목록을 반환합니다.

**Request**

```http
GET /intents HTTP/1.1
Host: localhost:1117
```

**Response**

```json
{
  "intents": [
    {
      "id": 0,
      "name": "메뉴 카테고리 안내",
      "description": "메뉴의 전체 카테고리를 안내합니다.",
      "examples": [
        "어떤 메뉴가 있나요?",
        "메뉴판 보여주세요"
      ]
    },
    {
      "id": 1,
      "name": "특정 상품 및 가격 안내",
      "description": "특정 메뉴의 가격을 안내합니다.",
      "examples": [
        "떡볶이 얼마예요?",
        "김밥 가격 알려주세요"
      ]
    },
    ...
  ],
  "total": 48
}
```

---

### 4. 메뉴 검색

#### `GET /menu/search`

메뉴를 검색합니다.

**Request**

```http
GET /menu/search?q=떡볶이&category=한식&limit=10 HTTP/1.1
Host: localhost:1117
```

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | 검색 키워드 |
| `category` | string | No | 카테고리 필터 |
| `limit` | integer | No | 결과 개수 제한 (기본값: 10) |
| `offset` | integer | No | 페이지네이션 오프셋 |

**Response**

```json
{
  "results": [
    {
      "menuId": "menu_001",
      "name": "떡볶이",
      "category": "한식",
      "price": 5000,
      "description": "매콤한 떡볶이",
      "tags": ["인기", "매운맛"],
      "available": true
    },
    {
      "menuId": "menu_002",
      "name": "치즈떡볶이",
      "category": "한식",
      "price": 6000,
      "description": "치즈가 들어간 떡볶이",
      "tags": ["신메뉴"],
      "available": true
    }
  ],
  "total": 2,
  "offset": 0,
  "limit": 10
}
```

---

### 5. 메뉴 상세 조회

#### `GET /menu/{menuId}`

특정 메뉴의 상세 정보를 조회합니다.

**Request**

```http
GET /menu/menu_001 HTTP/1.1
Host: localhost:1117
```

**Response**

```json
{
  "menuId": "menu_001",
  "name": "떡볶이",
  "category": "한식",
  "price": 5000,
  "description": "매콤한 떡볶이로 인기 메뉴입니다.",
  "tags": ["인기", "매운맛"],
  "available": true,
  "allergyInfo": ["밀", "대두"],
  "nutritionInfo": {
    "calories": 450,
    "protein": 8,
    "carbs": 82,
    "fat": 12
  },
  "imageUrl": "https://example.com/images/tteokbokki.jpg"
}
```

---

## Intent 목록

### 메뉴 관련 (0-10)

| ID | Intent Name | Description |
|----|-------------|-------------|
| 0 | 메뉴 카테고리 안내 | 전체 메뉴 카테고리 안내 |
| 1 | 특정 상품 및 가격 안내 | 특정 메뉴의 가격 정보 |
| 2 | 상품에 대한 상세 및 추가 안내 | 메뉴 상세 정보 |
| 3 | (뱃지) 인기메뉴 | 인기 메뉴 추천 |
| 4 | (뱃지) 추천메뉴 | 추천 메뉴 안내 |
| 5 | (뱃지) 대표메뉴 | 대표 메뉴 소개 |
| 6 | (뱃지) 할인, 이벤트 메뉴 | 할인/이벤트 메뉴 안내 |
| 7 | (뱃지) 1+1 메뉴 문의 | 1+1 프로모션 메뉴 |
| 8 | (뱃지) 신상 메뉴 | 신메뉴 안내 |
| 9 | (뱃지) 한정 메뉴 | 한정 판매 메뉴 |
| 10 | (뱃지) 매운맛 | 매운 메뉴 추천 |

### 주문 및 결제 (11-15)

| ID | Intent Name | Description |
|----|-------------|-------------|
| 11 | 매장 주문 방식 안내 | 키오스크, 테이블오더 등 |
| 12 | 주문한 상품 전달 방식 안내 | 내점, 포장, 배달 등 |
| 13 | 결제 방법 안내 | 현금, 카드, 간편결제 등 |
| 14 | 특정 결제 방법 상세 안내 | 특정 결제수단 상세 |
| 15 | 결제 방법 추가 안내 | 추가 결제 정보 |

### 영업 정보 (16-21)

| ID | Intent Name | Description |
|----|-------------|-------------|
| 16 | 영업 시간 안내 | 일반 영업시간 |
| 17 | 브레이크 타임 안내 | 브레이크 타임 정보 |
| 18 | 휴무일 안내 | 정기/임시 휴무 |
| 19 | 휴무일 상세 안내 | 휴무일 상세 |
| 20 | 영업시간 및 휴무일 추가 안내 | 추가 영업 정보 |
| 21 | 특정 요일에 대한 영업 여부 | 특정 요일 영업 확인 |

### 배달 및 위치 (22-29)

| ID | Intent Name | Description |
|----|-------------|-------------|
| 22 | 배달 가능 지역 안내 | 배달 가능 지역 |
| 23 | 배달비 및 최소 주문 금액 안내 | 배달비 정보 |
| 24 | 배달 추가 안내 | 배달 추가 정보 |
| 25 | 테이블 점유 안내 | 테이블 상황 |
| 26 | 테이블 점유 추가 안내 | 테이블 추가 정보 |
| 27 | 상점 주소 안내 및 지도 링크 전달 | 매장 위치 |
| 28 | 대중교통 이용 방법 안내 | 교통편 안내 |
| 29 | 근처 랜드마크 안내 | 주변 랜드마크 |

### 시설 정보 (30-33)

| ID | Intent Name | Description |
|----|-------------|-------------|
| 30 | 테이블 및 좌석 수 안내 | 좌석 정보 |
| 31 | 단체석 및 예약석 유무 안내 | 단체석 정보 |
| 32 | 야외 테라스 또는 개별 룸 여부 안내 | 특별 공간 정보 |
| 33 | 상점 규모 및 시설 추가 안내 | 시설 추가 정보 |

### 혜택 및 이벤트 (34-43)

| ID | Intent Name | Description |
|----|-------------|-------------|
| 34 | 멤버십 가입 안내 | 멤버십 가입 방법 |
| 35 | 멤버십 혜택 안내 | 멤버십 혜택 |
| 36 | 포인트 적립 안내 | 포인트 적립 |
| 37 | 포인트 사용 안내 | 포인트 사용 |
| 38 | 쿠폰 발행 안내 | 쿠폰 정보 |
| 39 | 쿠폰 사용 안내 | 쿠폰 사용 방법 |
| 40 | 멤버십, 포인트 및 쿠폰 추가 안내 | 혜택 추가 정보 |
| 41 | 현재 진행 중인 이벤트 안내 | 이벤트 정보 |
| 42 | 현재 진행 중인 이벤트 상세 안내 | 이벤트 상세 |
| 43 | 현재 진행 중인 이벤트 추가 안내 | 이벤트 추가 정보 |

### 기타 (44-47)

| ID | Intent Name | Description |
|----|-------------|-------------|
| 44 | CallBackIntent | 상담원 연결 요청 |
| 45 | 감사 인텐트 | 감사 표현 |
| 46 | 인사 인텐트 | 인사 |
| 47 | FallBackIntent | 이해 불가 |

---

## Entity Types

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| `B-MENU` | 메뉴명 시작 | 떡볶이, 김밥 |
| `I-MENU` | 메뉴명 계속 | 치즈떡볶이의 "치즈" |
| `B-PAYMENT` | 결제수단 시작 | 카드, 현금 |
| `I-PAYMENT` | 결제수단 계속 | 신용카드의 "신용" |
| `B-DAY` | 요일 | 월요일, 주말 |
| `O` | 기타 토큰 | 나머지 모든 토큰 |

---

## Error Handling

### Error Response Format

```json
{
  "statusCode": 400,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Input text is too long (max 500 characters)",
    "details": {
      "field": "body.text",
      "provided": 650,
      "max": 500
    }
  },
  "timestamp": "2025-01-02T10:30:00Z"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | 잘못된 입력 형식 |
| `MISSING_FIELD` | 400 | 필수 필드 누락 |
| `UNAUTHORIZED` | 401 | 인증 실패 |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate Limit 초과 |
| `MODEL_ERROR` | 500 | 모델 추론 오류 |
| `DATABASE_ERROR` | 500 | 데이터베이스 오류 |
| `SERVICE_UNAVAILABLE` | 503 | 서비스 일시 중단 |

---

## Rate Limiting

API 요청은 다음과 같이 제한됩니다:

- **기본 제한**: 100 requests/hour per IP
- **인증된 사용자**: 1000 requests/hour per API key

Rate Limit 정보는 응답 헤더에 포함됩니다:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1641038400
```

---

## Webhooks (Coming Soon)

특정 이벤트 발생 시 Webhook을 통해 알림을 받을 수 있습니다.

### Supported Events
- `order.completed`: 주문 완료
- `fallback.triggered`: Fallback 발생
- `error.occurred`: 에러 발생

---

## SDK & Libraries

### Python SDK

```bash
pip install beaver-ars-sdk
```

```python
from beaver_ars import BeaverARSClient

client = BeaverARSClient(api_key="your_api_key")

response = client.order(
    text="떡볶이 가격이 얼마예요?",
    user_id="user_001"
)

print(response.text)
print(response.intent)
print(response.entities)
```

### JavaScript SDK

```bash
npm install beaver-ars-sdk
```

```javascript
const BeaverARS = require('beaver-ars-sdk');

const client = new BeaverARS({ apiKey: 'your_api_key' });

client.order({
  text: '떡볶이 가격이 얼마예요?',
  userId: 'user_001'
}).then(response => {
  console.log(response.text);
  console.log(response.intent);
  console.log(response.entities);
});
```

---

## Changelog

### v1.0.0 (2025-01-02)
- Initial API release
- Intent Classification (48 classes)
- NER (6 entity types)
- Hybrid Search
- Template-based Response Generation

---

**문서 버전**: 1.0  
**최종 수정일**: 2025-01-02  
**Contact**: api-support@beaver-ars.com
