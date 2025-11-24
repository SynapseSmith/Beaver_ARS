# Production Deployment Guide

## 🚀 실제 현업 배포 프로세스

이 문서는 **Beaver ARS**를 실제 프로덕션 환경에 배포하는 전체 과정을 설명합니다.

---

## 📋 배포 체크리스트

### 1단계: 사전 준비 (Pre-Deployment)

- [ ] 코드 리뷰 완료
- [ ] 모든 테스트 통과 (`pytest tests/`)
- [ ] 보안 스캔 완료 (Bandit, Safety)
- [ ] 환경 변수 설정 확인 (`.env` 파일)
- [ ] 데이터베이스 백업 생성
- [ ] 배포 승인 획득

### 2단계: 빌드 & 테스트 (Build & Test)

- [ ] Docker 이미지 빌드
- [ ] 통합 테스트 실행
- [ ] 부하 테스트 실행
- [ ] 이미지 보안 스캔

### 3단계: 배포 (Deployment)

- [ ] Staging 환경 배포
- [ ] Staging 검증
- [ ] Production 배포
- [ ] Health check 확인
- [ ] 모니터링 대시보드 확인

### 4단계: 배포 후 (Post-Deployment)

- [ ] 로그 모니터링
- [ ] 성능 메트릭 확인
- [ ] 에러율 확인
- [ ] 사용자 피드백 수집

---

## 🔧 배포 파일 구조

```
Beaver_ARS_Portfolio/
├── 📦 Docker 설정
│   ├── Dockerfile                 # Multi-stage 빌드
│   ├── .dockerignore              # 빌드 제외 파일
│   └── docker-compose.yml         # 전체 스택 오케스트레이션
│
├── 🔄 CI/CD
│   └── .github/workflows/
│       └── ci-cd.yml              # 자동화된 배포 파이프라인
│
├── 🗄️ 데이터베이스
│   └── database/
│       ├── init.sql               # 스키마 & 초기 데이터
│       └── my.cnf                 # MySQL 설정
│
├── 🌐 웹 서버
│   ├── nginx/
│   │   └── nginx.conf             # Reverse proxy 설정
│   └── gunicorn_config.py         # WSGI 서버 설정
│
├── 📊 모니터링
│   └── monitoring/
│       ├── prometheus.yml         # 메트릭 수집
│       └── grafana/               # 대시보드 설정
│
├── 🔐 환경 변수
│   └── .env.example               # 환경 변수 템플릿
│
└── 🛠️ 배포 스크립트
    ├── deploy.sh                  # 자동 배포 스크립트
    ├── rollback.sh                # 롤백 스크립트
    └── scripts/
        ├── health_check.py        # 헬스 체크
        └── load_test.py           # 부하 테스트
```

---

## 🐳 Docker 배포

### 로컬 개발 환경

```bash
# 1. 환경 변수 설정
cp .env.example .env
nano .env

# 2. Docker Compose로 전체 스택 실행
docker-compose up -d

# 3. 로그 확인
docker-compose logs -f app

# 4. 헬스 체크
curl http://localhost:5000/health
```

### 프로덕션 환경

```bash
# 1. 이미지 빌드
docker-compose build --no-cache

# 2. 프로덕션 모드로 실행
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 3. 서비스 확인
docker-compose ps
```

---

## 🔄 CI/CD 파이프라인

### GitHub Actions 워크플로우

`.github/workflows/ci-cd.yml`에 정의된 자동화 프로세스:

#### 1️⃣ 코드 품질 검사 (Lint)
```yaml
- Black (코드 포맷팅)
- isort (import 정렬)
- Flake8 (린팅)
```

#### 2️⃣ 유닛 테스트
```yaml
- Python 3.8, 3.9, 3.10 매트릭스 테스트
- pytest with coverage
- Codecov 리포트
```

#### 3️⃣ 보안 스캔
```yaml
- Safety (의존성 취약점)
- Bandit (코드 보안 이슈)
```

#### 4️⃣ Docker 빌드 & 푸시
```yaml
- Multi-stage 빌드
- Docker Hub 푸시
- 이미지 캐싱
```

#### 5️⃣ 배포
```yaml
- SSH를 통한 서버 배포
- Health check 검증
- Slack 알림
```

### 배포 트리거

```bash
# main 브랜치에 푸시 시 자동 배포
git push origin main

# PR 생성 시 테스트만 실행
git push origin feature/new-feature
```

---

## 🚀 수동 배포 프로세스

### 1. 배포 스크립트 사용

```bash
# 서버에 SSH 접속
ssh user@your-server.com

# 배포 실행
cd /opt/beaver-ars
sudo ./deploy.sh
```

**deploy.sh가 수행하는 작업:**
1. ✅ 사전 요구사항 확인 (Docker, Git)
2. 📥 최신 코드 Pull
3. 💾 현재 상태 백업
4. 🔐 환경 변수 업데이트
5. 🐳 Docker 이미지 빌드
6. 🗄️ 데이터베이스 마이그레이션
7. 🚀 서비스 시작
8. 🏥 Health check 검증

### 2. 롤백 (문제 발생 시)

```bash
# 이전 버전으로 롤백
sudo ./rollback.sh

# 백업 목록에서 선택
# 자동으로 서비스 복원
```

---

## 📊 모니터링 & 로깅

### Prometheus + Grafana

```bash
# Grafana 대시보드 접속
http://your-server:3000

# 기본 계정
Username: admin
Password: (GRAFANA_PASSWORD in .env)
```

**모니터링 메트릭:**
- 📈 API 요청률 (requests/sec)
- ⏱️ 응답 시간 (p50, p95, p99)
- ❌ 에러율
- 💾 메모리 사용량
- 💿 디스크 사용량
- 🗄️ 데이터베이스 연결 수

### 로그 확인

```bash
# 애플리케이션 로그
docker-compose logs -f app

# Nginx 로그
docker-compose logs -f nginx

# 특정 시간대 로그
docker-compose logs --since 1h app

# 전체 로그 파일 위치
/opt/beaver-ars/logs/
```

---

## 🧪 배포 전 테스트

### 1. 유닛 테스트

```bash
pytest tests/ -v --cov=src
```

### 2. 통합 테스트

```bash
# API 엔드포인트 테스트
pytest tests/test_api_endpoints.py -v
```

### 3. 부하 테스트

```bash
# 100개 요청, 10개 동시 워커
python scripts/load_test.py --requests 100 --workers 10

# 커스텀 URL
python scripts/load_test.py --url http://your-server:5000 --requests 1000
```

### 4. 헬스 체크

```bash
# 전체 시스템 헬스 체크
python scripts/health_check.py

# API만 체크
curl http://localhost:5000/health
```

---

## 🔐 보안 체크리스트

### 배포 전

- [ ] `.env` 파일에 실제 패스워드 설정
- [ ] SECRET_KEY 변경
- [ ] JWT_SECRET_KEY 변경
- [ ] 데이터베이스 비밀번호 강화
- [ ] Redis 비밀번호 설정
- [ ] SSL 인증서 설치 (Let's Encrypt)
- [ ] 방화벽 규칙 설정
- [ ] 불필요한 포트 차단

### 런타임

- [ ] HTTPS 강제 (HTTP → HTTPS 리다이렉트)
- [ ] Rate limiting 활성화
- [ ] CORS 설정 확인
- [ ] Security headers 적용
- [ ] SQL Injection 방어
- [ ] XSS 방어

---

## 🌐 클라우드 배포 가이드

### AWS 배포

```bash
# 1. EC2 인스턴스 생성 (Ubuntu 22.04, t3.medium 이상)

# 2. Docker 설치
sudo apt-get update
sudo apt-get install docker.io docker-compose -y

# 3. 프로젝트 클론
git clone https://github.com/your-repo/beaver-ars.git
cd beaver-ars

# 4. 환경 변수 설정
cp .env.example .env
nano .env

# 5. 배포 실행
sudo ./deploy.sh

# 6. Nginx SSL 설정 (Let's Encrypt)
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### GCP 배포

```bash
# 1. Compute Engine VM 생성

# 2. Docker 설치 및 배포 (AWS와 동일)

# 3. Load Balancer 설정
# - Cloud Load Balancing 콘솔에서 설정
# - Health check 경로: /health
```

---

## 📈 성능 최적화

### 1. 데이터베이스 최적화

```sql
-- 인덱스 추가 (init.sql에 포함)
CREATE INDEX idx_orders_user_time ON orders(user_id, order_time);

-- 쿼리 캐시 활성화 (my.cnf)
query_cache_type = 1
query_cache_size = 64M
```

### 2. Redis 캐싱

```python
# 자주 조회되는 데이터 캐싱
# - 메뉴 정보
# - FAQ 답변
# - 인기 메뉴 순위
```

### 3. Nginx 최적화

```nginx
# gzip 압축 (nginx.conf에 포함)
gzip on;
gzip_comp_level 6;

# 정적 파일 캐싱
location /static/ {
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

---

## 🆘 트러블슈팅

### 서비스가 시작되지 않는 경우

```bash
# 로그 확인
docker-compose logs app

# 컨테이너 상태 확인
docker-compose ps

# 포트 충돌 확인
sudo netstat -tulpn | grep :5000
```

### 데이터베이스 연결 실패

```bash
# MySQL 컨테이너 확인
docker-compose logs mysql

# 직접 접속 테스트
docker-compose exec mysql mysql -uroot -p
```

### Health check 실패

```bash
# Health check 스크립트 실행
python scripts/health_check.py

# 각 서비스 개별 확인
curl http://localhost:5000/health
```

---

## 📞 지원 & 문의

배포 중 문제가 발생하면:

1. **로그 확인**: `docker-compose logs -f`
2. **헬스 체크**: `python scripts/health_check.py`
3. **롤백**: `sudo ./rollback.sh`
4. **GitHub Issues**: 문제 리포팅

---

## ✅ 배포 완료 확인

배포가 성공적으로 완료되면 다음을 확인하세요:

- ✅ http://your-server:5000/health → 200 OK
- ✅ Grafana 대시보드에 메트릭 표시
- ✅ 로그에 에러 없음
- ✅ API 응답 시간 < 500ms
- ✅ CPU/메모리 사용량 정상 범위

---

**🎉 축하합니다! 프로덕션 배포가 완료되었습니다!**
