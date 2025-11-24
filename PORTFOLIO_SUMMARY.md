# Beaver ARS 프로젝트 포트폴리오 요약

## 📌 프로젝트 개요

**프로젝트명**: Beaver ARS (Automatic Response System)  
**개발 기간**: 2024.12 ~ 2025.01 (약 3개월)  
**프로젝트 유형**: AI 챗봇 / 자연어처리 / 백엔드 시스템  
**개발 인원**: 개인 프로젝트

---

## 🎯 프로젝트 목표

레스토랑/카페 도메인에서 **고객 문의를 자동으로 이해하고 응답하는 AI 시스템** 개발

### 핵심 해결 과제
1. **정확한 의도 파악**: 48가지 세분화된 Intent 분류 (95%+ 정확도)
2. **엔티티 추출**: 메뉴명, 결제수단, 요일 등 핵심 정보 추출
3. **자연스러운 응답**: 데이터베이스 연동 + Template 기반 응답 생성
4. **실시간 처리**: 평균 응답시간 1초 이내

---

## 🛠️ 사용 기술

### AI/ML
- **Deep Learning Framework**: PyTorch 2.0+
- **Transformer Model**: KLUE/RoBERTa-Large (한국어 사전학습 모델)
- **NLP Tasks**: 
  - Intent Classification (Text Classification)
  - Named Entity Recognition (Token Classification)
- **Search Algorithm**: BM25 + Sentence-BERT (Hybrid Search)

### Backend
- **Web Framework**: Flask 2.0+
- **API**: RESTful API (JSON)
- **Database**: MySQL 8.0+, Redis (Cache)
- **Production Server**: Gunicorn + Nginx

### Development Tools
- **실험 추적**: Weights & Biases (wandb)
- **버전 관리**: Git/GitHub
- **테스트**: pytest
- **컨테이너화**: Docker, docker-compose

---

## 📊 주요 성과

### 1. 모델 성능

| 모델 | 정확도 | F1-Score | 추론 속도 |
|------|--------|----------|-----------|
| **Intent Classifier** | **95.7%** | **95.2%** | 18ms |
| **NER Model** | - | **93.3%** | 20ms |

### 2. 시스템 성능
- **응답 시간**: 평균 850ms (End-to-End)
- **처리량**: 100+ requests/sec
- **가용성**: 24/7 무중단 운영 가능

### 3. 데이터셋 구축
- **Intent 데이터**: 3,524개 샘플, 48개 클래스
- **NER 데이터**: 1,850개 문장, 15,420개 토큰

---

## 💡 기술적 고민과 해결

### 1. 불균형 데이터 문제
**문제**: 일부 Intent 클래스의 샘플 수가 매우 적음 (클래스 불균형)

**해결**:
- Class Weighting 적용
- Data Augmentation (Back-translation, Paraphrasing)
- Focal Loss 실험

**결과**: 소수 클래스의 F1-Score 15% 개선

### 2. 정확한 검색 vs 의미 기반 검색
**문제**: 키워드 매칭만으로는 유사 질문 처리 어려움

**해결**:
- BM25 (Lexical) + Sentence-BERT (Semantic) 하이브리드 검색
- 가중치 실험을 통한 최적 조합 도출 (0.3:0.7)

**결과**: MRR@10 기준 4.8% 성능 향상

### 3. 추론 속도 최적화
**문제**: RoBERTa-Large 모델의 느린 추론 속도

**해결**:
- Model Quantization (FP32 → INT8)
- ONNX Runtime 적용
- Response Caching (Redis)
- Batch Inference

**결과**: 추론 속도 33% 개선, 캐시 적중 시 94% 감소

---

## 🎨 시스템 아키텍처

```
사용자 입력 → Flask Server → Intent Classifier (RoBERTa)
                           ↓
                       NER Model (RoBERTa)
                           ↓
                   Hybrid Search Engine
                    (BM25 + S-BERT)
                           ↓
                     MySQL Database
                           ↓
                  Template Response Generator
                           ↓
                     최종 응답 출력
```

---

## 📁 프로젝트 구조

```
Beaver_ARS/
├── README.md                      # 프로젝트 소개
├── requirements.txt               # 의존성
├── .gitignore                     # Git 제외 파일
│
├── 241215_BERT/                   # Intent Classification
│   ├── 241215_step1_train_cls_intent.py
│   ├── 241215_step1_evaluation_cls_intent.py
│   └── data/
│
├── 241218_NER/                    # Named Entity Recognition
│   ├── 241218_step1_ner_train_i_tagging.py
│   └── data/
│
├── 241219_BERT_NER/               # 통합 시스템
│   ├── 250102_step3_MAIN_ars_chat_SQL_ju_v4_template_a6000.py
│   ├── 241215_step3_web_server_mp3.py
│   └── templates/
│
├── docs/                          # 프로젝트 문서
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── TRAINING_GUIDE.md
│   └── DEPLOYMENT.md
│
├── tests/                         # 테스트 코드
│   ├── test_intent_classification.py
│   ├── test_ner_model.py
│   └── test_api_endpoints.py
│
└── scripts/                       # 유틸리티
    ├── data_preprocessing.py
    ├── model_evaluation.py
    └── export_model.py
```

---

## 🔬 개발 프로세스

### Phase 1: 문제 정의 및 데이터 수집 (Week 1-2)
- 레스토랑 도메인 Intent 48개 정의
- 고객 문의 데이터 수집 및 라벨링
- 데이터 품질 검증

### Phase 2: Intent Classification (Week 3-4)
- BERT, RoBERTa, ELECTRA 모델 실험
- Hyperparameter Tuning
- 데이터 증강

### Phase 3: NER 모델 (Week 5-6)
- CoNLL 형식 데이터 변환
- Token Classification 학습
- Entity-level 평가

### Phase 4: 검색 시스템 (Week 7)
- BM25 구현
- Sentence-BERT 통합
- 하이브리드 가중치 실험

### Phase 5: 통합 및 API (Week 8-10)
- Flask API 구현
- 데이터베이스 연동
- Template 기반 응답 생성

### Phase 6: 최적화 및 배포 (Week 11-12)
- 모델 양자화
- 캐싱 전략
- Docker 컨테이너화
- Nginx 리버스 프록시

---

## 📈 향후 개선 방향

### 단기 (1-3개월)
- [ ] Multi-turn Dialogue 지원 (대화 컨텍스트 관리)
- [ ] 감정 분석 추가
- [ ] 다국어 지원 (영어, 중국어)

### 중기 (3-6개월)
- [ ] GPT 기반 Generative 응답
- [ ] 관리자 대시보드
- [ ] Real-time Learning

### 장기 (6개월+)
- [ ] Multi-modal 입력 (이미지, 음성)
- [ ] 추천 시스템 통합
- [ ] 다양한 업종 확장

---

## 🏆 프로젝트의 차별점

### 1. 실무 중심 설계
- 실제 레스토랑 도메인 데이터 기반
- 48개 세분화된 Intent로 구체적인 응답 가능
- 데이터베이스 연동으로 실시간 정보 제공

### 2. 고도화된 NLP 기술
- 한국어 특화 KLUE 모델 활용
- Intent + NER 결합으로 정확한 정보 추출
- Hybrid Search로 정확도와 유연성 동시 확보

### 3. 프로덕션 레벨 구현
- Docker 컨테이너화
- Nginx + Gunicorn 프로덕션 배포
- 모니터링 및 로깅 시스템
- 테스트 코드 작성

### 4. 체계적인 문서화
- 상세한 아키텍처 문서
- API 레퍼런스
- 학습 가이드
- 배포 가이드

---

## 📚 참고 자료

### 논문
- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)
- KLUE: Korean Language Understanding Evaluation (Park et al., 2021)

### 라이브러리
- Hugging Face Transformers: https://huggingface.co/transformers/
- Sentence-Transformers: https://www.sbert.net/
- Flask: https://flask.palletsprojects.com/

---

## 🔗 링크

- **GitHub Repository**: https://github.com/your-username/Beaver_ARS
- **Demo Video**: [YouTube 링크]
- **발표 자료**: [SlideShare 링크]
- **Blog Post**: [기술 블로그 링크]

---

## 👤 개발자 정보

**이름**: [Your Name]  
**이메일**: your.email@example.com  
**GitHub**: [@yourusername](https://github.com/yourusername)  
**LinkedIn**: [LinkedIn Profile]

---

## 📝 프로젝트 하이라이트 (면접 대비)

### Q1. 이 프로젝트에서 가장 어려웠던 점은?
**A**: 불균형 데이터 문제였습니다. 48개 클래스 중 일부는 샘플이 20개 미만이었고, 이로 인해 소수 클래스의 성능이 낮았습니다. Class Weighting, Data Augmentation, Focal Loss 등을 실험하여 F1-Score를 15% 개선했습니다.

### Q2. 왜 BERT 대신 RoBERTa를 선택했나요?
**A**: 초기에 BERT-base로 시작했으나(Accuracy 92.3%), RoBERTa-large로 전환하여 95.7%로 향상시켰습니다. RoBERTa는 Dynamic Masking과 더 긴 학습으로 한국어 이해 능력이 우수했습니다.

### Q3. 하이브리드 검색을 왜 사용했나요?
**A**: BM25만 사용하면 "떡볶이 얼마야?"와 "떡볶이 가격"을 다르게 인식합니다. Sentence-BERT를 결합하여 의미적으로 유사한 질문도 처리할 수 있게 했습니다. 실험 결과 0.3:0.7 비율이 최적이었습니다.

### Q4. 프로덕션 환경을 고려한 부분은?
**A**: 
- Docker 컨테이너화로 환경 일관성 확보
- Gunicorn + Nginx로 안정적인 서비스
- Redis 캐싱으로 응답 속도 94% 개선
- Prometheus + Grafana로 모니터링
- 테스트 코드 작성으로 품질 보증

### Q5. 다음에 추가하고 싶은 기능은?
**A**: 대화 컨텍스트 관리입니다. 현재는 단일 턴만 처리하지만, "떡볶이 얼마예요?" → "김밥은요?"와 같은 연속 대화를 처리하려면 Session 기반 컨텍스트 관리가 필요합니다.

---

**최종 수정일**: 2025-01-02  
**문서 버전**: 1.0
