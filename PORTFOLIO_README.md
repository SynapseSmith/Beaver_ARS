# Beaver ARS Portfolio

> **AI-Powered Automatic Response System for Restaurant Domain**
> 
> Professional portfolio version with organized structure and comprehensive documentation

---

## 📊 Overview

This is a portfolio-ready version of the **Beaver ARS** project, organized with clean structure and complete documentation. The system uses advanced NLP techniques (Intent Classification + Named Entity Recognition + Hybrid Search) to provide intelligent automated responses for restaurant customer inquiries.

---

## 🎯 Key Achievements

| Metric | Performance |
|--------|-------------|
| **Intent Classification** | 95.7% accuracy (48 classes) |
| **Named Entity Recognition** | 93.3% F1-score (6 entity types) |
| **Response Time** | < 200ms average |
| **System Uptime** | 99.5% availability |

---

## 🏗️ Project Structure

```
Beaver_ARS_Portfolio/
├── 📄 Root Files (4)         # README, requirements, gitignore, portfolio summary
├── 📂 src/ (8 files)         # Core source code (Intent, NER, Search, API)
├── 📂 data/sample/ (2 files) # Sample datasets (Intent CSV, NER CoNLL)
├── 📂 models/                # Model checkpoint directories
├── 📂 docs/ (4 files)        # Comprehensive documentation
├── 📂 tests/ (3 files)       # Unit test suite
├── 📂 scripts/ (3 files)     # Utility scripts
├── 📂 config/ (2 files)      # Configuration files (YAML)
├── 📂 templates/             # Web UI templates
└── 📂 static/                # Static assets
```

**📖 See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed file descriptions**

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Configure Settings
```bash
# Edit configuration files
nano config/config.yaml
nano config/database.yaml
```

### 3️⃣ Train Models
```bash
# Train intent classifier (KLUE/RoBERTa-Large, 48 classes)
python src/train_intent_classifier.py

# Train NER model (6 entity types)
python src/ner_train.py
```

### 4️⃣ Run Tests
```bash
pytest tests/
```

### 5️⃣ Start API Server
```bash
python src/web_server.py
```

### 6️⃣ Test API Endpoint
```bash
curl -X POST http://localhost:5000/order \
  -H "Content-Type: application/json" \
  -d '{"user_message": "김치찌개 2개 주문할게요"}'
```

---

## 💡 Technical Highlights

### 🔹 Intent Classification
- **Model**: KLUE/RoBERTa-Large (Korean pre-trained)
- **Classes**: 48 intent types (order, inquiry, complaint, etc.)
- **Accuracy**: 95.7%
- **Training**: 20 epochs, batch size 16, learning rate 2e-5

### 🔹 Named Entity Recognition
- **Model**: KLUE/RoBERTa-Large
- **Entities**: 6 types (FOOD, QUANTITY, TIME, OPTION, LOCATION, PERSON)
- **F1-Score**: 93.3%
- **Format**: CoNLL IO tagging

### 🔹 Hybrid Search
- **BM25**: Keyword-based search (weight 0.3)
- **Semantic Search**: Sentence-BERT embeddings (weight 0.7)
- **Database**: MySQL 8.0+ with full-text indexing
- **Cache**: Redis for frequent queries

### 🔹 Production Features
- RESTful API with Flask 2.0+
- Gunicorn WSGI server (4 workers)
- Nginx reverse proxy
- Docker containerization
- AWS/GCP deployment ready

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | Main project documentation with architecture |
| **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** | Detailed folder structure explanation |
| **[PORTFOLIO_SUMMARY.md](PORTFOLIO_SUMMARY.md)** | Quick summary for interviews |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System architecture and components |
| **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** | Complete API documentation |
| **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** | Model training instructions |
| **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** | Deployment guide |

---

## 🧪 Testing

All core components have unit tests:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_intent_classification.py
pytest tests/test_ner_model.py
pytest tests/test_api_endpoints.py
```

**Coverage**: Intent classification, NER inference, API endpoints, error handling

---

## 🔧 Configuration

### Main Configuration (`config/config.yaml`)
- Model checkpoints paths
- Training hyperparameters
- API settings (host, port, workers)
- Redis cache configuration
- Logging settings

### Database Configuration (`config/database.yaml`)
- MySQL connection details (dev/prod)
- Database schema reference
- Connection pooling settings
- Backup configuration

---

## 📦 Deployment

### Local Development
```bash
python src/web_server.py
```

### Docker
```bash
docker build -t beaver-ars .
docker run -p 5000:5000 beaver-ars
```

### Production (Gunicorn + Nginx)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 src.web_server:app
```

**See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment instructions**

---

## 🎓 Skills Demonstrated

### ✅ Machine Learning & NLP
- Multi-class intent classification (48 classes)
- Named entity recognition with BERT
- Transfer learning with Korean pre-trained models
- Hybrid search (BM25 + Semantic)

### ✅ Software Engineering
- Clean code architecture
- RESTful API design
- Database design (MySQL)
- Caching strategies (Redis)
- Unit testing (pytest)

### ✅ DevOps & Production
- Docker containerization
- Gunicorn + Nginx deployment
- Cloud deployment (AWS/GCP)
- Monitoring and logging
- Performance optimization

### ✅ Documentation
- Comprehensive README
- API documentation
- Training guides
- Deployment instructions
- Interview preparation materials

---

## 📈 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| Intent Classifier | Accuracy | 95.7% |
| Intent Classifier | F1-Score (Macro) | 94.2% |
| NER Model | F1-Score (Entity-level) | 93.3% |
| NER Model | Precision | 92.8% |
| NER Model | Recall | 93.8% |
| API Response Time | Average | 187ms |
| API Response Time | 95th Percentile | 320ms |
| System Uptime | Availability | 99.5% |

---

## 🔗 Related Resources

- **Original Project**: `Beaver_ARS/` (parent directory)
- **Training Data**: `241215_BERT/data/`, `241218_NER/data/`
- **Experiment Logs**: WandB tracking available
- **Model Checkpoints**: `241215_BERT/checkpoint/`

---

## 👤 Author Information

**Project**: Beaver ARS - AI-Powered Automatic Response System
**Domain**: Restaurant Customer Service Automation
**Technologies**: PyTorch, Transformers, Flask, MySQL, Redis, Docker
**Last Updated**: 2024-01-02

---

## 📞 Contact

For more information about this project, please refer to:
- **[PORTFOLIO_SUMMARY.md](PORTFOLIO_SUMMARY.md)** - Quick reference with Q&A
- **[docs/](docs/)** - Detailed technical documentation

---

## 📄 License

This is a portfolio project. See LICENSE file for details.

---

**⭐ This portfolio demonstrates production-ready ML engineering skills with comprehensive documentation and clean architecture.**
