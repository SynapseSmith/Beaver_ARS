# 배포 가이드

## 목차
1. [개발 환경 배포](#개발-환경-배포)
2. [프로덕션 배포](#프로덕션-배포)
3. [Docker 배포](#docker-배포)
4. [클라우드 배포](#클라우드-배포)
5. [모니터링 설정](#모니터링-설정)
6. [문제 해결](#문제-해결)

---

## 개발 환경 배포

### 1. 로컬 개발 서버

```bash
# 가상환경 활성화
source venv/bin/activate

# Flask 개발 서버 실행
cd 241219_BERT_NER
python 241215_step3_web_server_mp3.py

# 별도 터미널에서 메인 API 서버 실행
python 250102_step3_MAIN_ars_chat_SQL_ju_v4_template_a6000.py
```

**접속**
- 웹 UI: http://localhost:5007
- API: http://localhost:1117

### 2. 환경 변수 설정

`.env` 파일 생성:
```bash
# Flask 설정
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1

# 모델 경로
INTENT_MODEL_PATH=checkpoint/klue_roberta_large_v9
NER_MODEL_PATH=241218_NER/ner_checkpoint2

# 데이터베이스
DATABASE_URL=mysql://user:password@localhost:3306/beaver_ars
REDIS_URL=redis://localhost:6379/0

# API 키
API_KEY=your_secret_api_key_here

# 로깅
LOG_LEVEL=INFO
LOG_DIR=logs/
```

---

## 프로덕션 배포

### 1. Gunicorn (WSGI 서버)

#### 설치
```bash
pip install gunicorn
```

#### 실행
```bash
gunicorn \
    --bind 0.0.0.0:1117 \
    --workers 4 \
    --threads 2 \
    --timeout 120 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    app:app
```

#### Gunicorn 설정 파일 (`gunicorn_config.py`)

```python
import multiprocessing

# 서버 소켓
bind = "0.0.0.0:1117"
backlog = 2048

# Worker 프로세스
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# 로깅
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 프로세스 naming
proc_name = "beaver_ars"

# 재시작
max_requests = 1000
max_requests_jitter = 50

# SSL (HTTPS 사용 시)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
```

실행:
```bash
gunicorn -c gunicorn_config.py app:app
```

### 2. Nginx (리버스 프록시)

#### 설치
```bash
sudo apt-get update
sudo apt-get install nginx
```

#### Nginx 설정 (`/etc/nginx/sites-available/beaver-ars`)

```nginx
upstream beaver_ars_api {
    server 127.0.0.1:1117;
}

upstream beaver_ars_web {
    server 127.0.0.1:5007;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # API 엔드포인트
    location /api/ {
        proxy_pass http://beaver_ars_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout 설정
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        
        # Buffer 설정
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # 웹 UI
    location / {
        proxy_pass http://beaver_ars_web/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # 정적 파일
    location /static/ {
        alias /path/to/Beaver_ARS/static/;
        expires 30d;
    }
    
    # 로그
    access_log /var/log/nginx/beaver_ars_access.log;
    error_log /var/log/nginx/beaver_ars_error.log;
}
```

#### SSL 설정 (Let's Encrypt)

```bash
# Certbot 설치
sudo apt-get install certbot python3-certbot-nginx

# SSL 인증서 발급
sudo certbot --nginx -d your-domain.com

# 자동 갱신 설정
sudo certbot renew --dry-run
```

#### Nginx 재시작

```bash
sudo ln -s /etc/nginx/sites-available/beaver-ars /etc/nginx/sites-enabled/
sudo nginx -t  # 설정 테스트
sudo systemctl restart nginx
```

### 3. Systemd 서비스

#### API 서비스 (`/etc/systemd/system/beaver-ars-api.service`)

```ini
[Unit]
Description=Beaver ARS API Server
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/path/to/Beaver_ARS
Environment="PATH=/path/to/Beaver_ARS/venv/bin"
ExecStart=/path/to/Beaver_ARS/venv/bin/gunicorn -c gunicorn_config.py app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 웹 서비스 (`/etc/systemd/system/beaver-ars-web.service`)

```ini
[Unit]
Description=Beaver ARS Web Server
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/path/to/Beaver_ARS/241219_BERT_NER
Environment="PATH=/path/to/Beaver_ARS/venv/bin"
ExecStart=/path/to/Beaver_ARS/venv/bin/python 241215_step3_web_server_mp3.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 서비스 활성화

```bash
sudo systemctl daemon-reload
sudo systemctl enable beaver-ars-api
sudo systemctl enable beaver-ars-web
sudo systemctl start beaver-ars-api
sudo systemctl start beaver-ars-web

# 상태 확인
sudo systemctl status beaver-ars-api
sudo systemctl status beaver-ars-web
```

---

## Docker 배포

### 1. Dockerfile

```dockerfile
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

# 포트 노출
EXPOSE 1117 5007

# 환경 변수
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 실행
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
```

### 2. docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: beaver-ars-api
    ports:
      - "1117:1117"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=mysql://user:password@db:3306/beaver_ars
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./logs:/app/logs
      - ./checkpoint:/app/checkpoint
    depends_on:
      - db
      - redis
    restart: always
  
  web:
    build: .
    container_name: beaver-ars-web
    ports:
      - "5007:5007"
    command: python 241219_BERT_NER/241215_step3_web_server_mp3.py
    volumes:
      - ./logs:/app/logs
      - ./static:/app/static
    depends_on:
      - api
    restart: always
  
  db:
    image: mysql:8.0
    container_name: beaver-ars-db
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: beaver_ars
      MYSQL_USER: user
      MYSQL_PASSWORD: password
    volumes:
      - mysql_data:/var/lib/mysql
    restart: always
  
  redis:
    image: redis:7-alpine
    container_name: beaver-ars-redis
    volumes:
      - redis_data:/data
    restart: always
  
  nginx:
    image: nginx:alpine
    container_name: beaver-ars-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - web
    restart: always

volumes:
  mysql_data:
  redis_data:
```

### 3. Docker 빌드 및 실행

```bash
# 빌드
docker-compose build

# 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

---

## 클라우드 배포

### AWS EC2

#### 1. EC2 인스턴스 생성
- **인스턴스 타입**: t3.xlarge (최소), g4dn.xlarge (GPU 필요 시)
- **AMI**: Ubuntu 22.04 LTS
- **스토리지**: 100GB SSD
- **보안 그룹**: 
  - SSH (22)
  - HTTP (80)
  - HTTPS (443)
  - Custom TCP (1117, 5007)

#### 2. 배포 스크립트

```bash
#!/bin/bash

# 패키지 업데이트
sudo apt-get update
sudo apt-get upgrade -y

# Python 설치
sudo apt-get install -y python3.9 python3-pip python3-venv

# 프로젝트 클론
git clone https://github.com/your-repo/Beaver_ARS.git
cd Beaver_ARS

# 가상환경 설정
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집

# Nginx 설치 및 설정
sudo apt-get install -y nginx
sudo cp nginx.conf /etc/nginx/sites-available/beaver-ars
sudo ln -s /etc/nginx/sites-available/beaver-ars /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# 서비스 등록
sudo cp beaver-ars-api.service /etc/systemd/system/
sudo cp beaver-ars-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable beaver-ars-api
sudo systemctl enable beaver-ars-web
sudo systemctl start beaver-ars-api
sudo systemctl start beaver-ars-web
```

### Google Cloud Platform (GCP)

#### Cloud Run 배포

```bash
# gcloud CLI 설치 및 인증
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Container Registry에 이미지 푸시
docker build -t gcr.io/YOUR_PROJECT_ID/beaver-ars:latest .
docker push gcr.io/YOUR_PROJECT_ID/beaver-ars:latest

# Cloud Run 배포
gcloud run deploy beaver-ars \
    --image gcr.io/YOUR_PROJECT_ID/beaver-ars:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 120
```

---

## 모니터링 설정

### 1. Prometheus

#### prometheus.yml
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'beaver-ars'
    static_configs:
      - targets: ['localhost:1117']
```

### 2. Grafana 대시보드

```bash
# Grafana 설치
sudo apt-get install -y grafana

# Prometheus 데이터소스 추가
# http://localhost:3000

# 주요 메트릭
# - Request Rate (requests/sec)
# - Response Time (ms)
# - Error Rate (%)
# - CPU/Memory Usage
```

### 3. 로그 수집 (ELK Stack)

```bash
# Elasticsearch, Logstash, Kibana 설치
docker-compose -f elk-docker-compose.yml up -d
```

---

## 문제 해결

### 서버가 시작되지 않음

```bash
# 로그 확인
sudo journalctl -u beaver-ars-api -f

# 포트 사용 확인
sudo netstat -tulpn | grep 1117

# 프로세스 확인
ps aux | grep gunicorn
```

### 메모리 부족

```bash
# 스왑 메모리 추가
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### SSL 인증서 문제

```bash
# 인증서 갱신
sudo certbot renew

# Nginx 재시작
sudo systemctl restart nginx
```

---

**문서 버전**: 1.0  
**최종 수정일**: 2025-01-02
