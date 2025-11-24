# Beaver ARS Portfolio - Project Structure Documentation

## 📁 Directory Structure

```
Beaver_ARS_Portfolio/
├── 📄 README.md                          # Main project documentation
├── 📄 PORTFOLIO_SUMMARY.md               # Quick summary for interviews
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore rules
│
├── 📂 src/                               # Source code directory
│   ├── intent_mapping.py                 # Intent label definitions (48 classes)
│   ├── train_intent_classifier.py        # Intent classification training
│   ├── inference_intent_classifier.py    # Intent classification inference
│   ├── evaluation_intent_classifier.py   # Intent model evaluation
│   ├── ner_train.py                      # NER model training (6 entity types)
│   ├── ner_evaluation.py                 # NER model evaluation
│   ├── main_system.py                    # Integrated ARS system (Intent + NER + Search)
│   └── web_server.py                     # Flask web server for API
│
├── 📂 data/                              # Data directory
│   └── sample/                           # Sample datasets
│       ├── intent_sample.csv             # Sample intent classification data
│       └── ner_sample.conll              # Sample NER training data (CoNLL format)
│
├── 📂 models/                            # Model checkpoints directory
│   ├── intent_classifier/                # Intent classification model checkpoints
│   │   └── (Place trained models here)
│   └── ner_model/                        # NER model checkpoints
│       └── (Place trained models here)
│
├── 📂 docs/                              # Documentation directory
│   ├── ARCHITECTURE.md                   # System architecture details
│   ├── API_REFERENCE.md                  # REST API documentation
│   ├── TRAINING_GUIDE.md                 # Model training instructions
│   └── DEPLOYMENT.md                     # Deployment guide (Docker, AWS, GCP)
│
├── 📂 tests/                             # Test suite directory
│   ├── test_intent_classification.py     # Intent classification tests
│   ├── test_ner_model.py                 # NER model tests
│   └── test_api_endpoints.py             # API endpoint tests
│
├── 📂 scripts/                           # Utility scripts directory
│   ├── data_preprocessing.py             # Data validation and preprocessing
│   ├── model_evaluation.py               # Model evaluation utilities
│   └── export_model.py                   # Model export (ONNX, TorchScript)
│
├── 📂 config/                            # Configuration directory
│   ├── config.yaml                       # Main configuration file
│   └── database.yaml                     # Database configuration
│
├── 📂 templates/                         # Web UI templates
│   └── (HTML templates for web interface)
│
└── 📂 static/                            # Static files (CSS, JS, images)
    └── (Static assets for web interface)
```

---

## 📋 File Descriptions

### Root Files

| File | Description |
|------|-------------|
| `README.md` | Comprehensive project overview with architecture diagrams, features, installation guide |
| `PORTFOLIO_SUMMARY.md` | Quick reference for interviews with key achievements and Q&A |
| `requirements.txt` | All Python dependencies with versions (PyTorch, Transformers, Flask, etc.) |
| `.gitignore` | Git ignore patterns for checkpoints, logs, data files |

### Source Code (`src/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `intent_mapping.py` | Intent label definitions | 48 intent classes mapping |
| `train_intent_classifier.py` | Intent model training | KLUE/RoBERTa-Large, 20 epochs, WandB tracking |
| `inference_intent_classifier.py` | Intent prediction | Real-time inference with confidence scores |
| `evaluation_intent_classifier.py` | Intent model evaluation | Accuracy, F1-score, confusion matrix |
| `ner_train.py` | NER model training | 6 entity types (FOOD, QUANTITY, TIME, etc.) |
| `ner_evaluation.py` | NER model evaluation | Entity-level F1-score, precision, recall |
| `main_system.py` | Integrated ARS system | Intent + NER + Hybrid Search (BM25 + Semantic) |
| `web_server.py` | Flask API server | RESTful endpoints, MySQL integration, Redis caching |

### Data (`data/sample/`)

| File | Format | Description |
|------|--------|-------------|
| `intent_sample.csv` | CSV | Sample intent classification dataset with user queries and labels |
| `ner_sample.conll` | CoNLL | Sample NER dataset in CoNLL format (token-level annotations) |

### Models (`models/`)

- **intent_classifier/**: Trained intent classification model checkpoints
  - `best_model.pt` - Best performing model from training
  - `config.json` - Model configuration
  - `tokenizer/` - KLUE tokenizer files

- **ner_model/**: Trained NER model checkpoints
  - `best_model.pt` - Best performing NER model
  - `label_map.json` - Entity label mapping
  - `tokenizer/` - KLUE tokenizer files

### Documentation (`docs/`)

| File | Contents |
|------|----------|
| `ARCHITECTURE.md` | System components, data flow, microservices, optimization strategies |
| `API_REFERENCE.md` | REST API endpoints, request/response formats, 48 intent definitions |
| `TRAINING_GUIDE.md` | Data preparation, hyperparameters, training pipeline, evaluation |
| `DEPLOYMENT.md` | Local setup, Docker containerization, cloud deployment (AWS/GCP) |

### Tests (`tests/`)

| Test File | Coverage |
|-----------|----------|
| `test_intent_classification.py` | Intent model loading, inference, accuracy validation |
| `test_ner_model.py` | NER model loading, entity extraction, F1-score validation |
| `test_api_endpoints.py` | API request/response validation, error handling |

### Scripts (`scripts/`)

| Script | Functionality |
|--------|---------------|
| `data_preprocessing.py` | CSV/CoNLL validation, data cleaning, train/test split |
| `model_evaluation.py` | Confusion matrix plotting, classification reports |
| `export_model.py` | Model export to ONNX/TorchScript for production |

### Configuration (`config/`)

| File | Purpose |
|------|---------|
| `config.yaml` | Model configs, training hyperparameters, API settings, logging |
| `database.yaml` | MySQL connection details, table schemas, backup configuration |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Settings
```bash
# Edit configuration files
nano config/config.yaml
nano config/database.yaml
```

### 3. Train Models
```bash
# Train intent classifier
python src/train_intent_classifier.py

# Train NER model
python src/ner_train.py
```

### 4. Run Tests
```bash
pytest tests/
```

### 5. Start API Server
```bash
python src/web_server.py
```

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Intent Classes** | 48 |
| **Intent Accuracy** | 95.7% |
| **NER Entity Types** | 6 |
| **NER F1-Score** | 93.3% |
| **Code Files** | 15 |
| **Documentation Pages** | 5 |
| **Test Files** | 3 |
| **Utility Scripts** | 3 |

---

## 🔗 Related Files

- **Original Project**: Located in parent `Beaver_ARS/` directory
- **Training Data**: Original datasets in `241215_BERT/data/` and `241218_NER/data/`
- **Experiment Logs**: WandB logs in `241215_BERT/wandb/` and `241218_NER/wandb/`
- **Model Checkpoints**: Original checkpoints in `241215_BERT/checkpoint/`

---

## 📝 Notes

1. **Model Checkpoints**: Place trained model files in `models/intent_classifier/` and `models/ner_model/` directories
2. **Database Setup**: Run MySQL schema creation scripts before starting the API server
3. **Configuration**: Update `config/database.yaml` with your MySQL credentials
4. **Environment Variables**: Consider using `.env` file for sensitive information in production
5. **Testing**: Ensure all tests pass before deployment (`pytest tests/`)

---

## 🎯 Portfolio Highlights

This organized structure demonstrates:

✅ **Clean Code Organization**: Logical separation of concerns (src, data, models, docs)
✅ **Comprehensive Documentation**: 5 detailed documentation files covering all aspects
✅ **Testing Coverage**: Unit tests for all major components
✅ **Production-Ready**: Configuration management, error handling, deployment guides
✅ **Scalability**: Modular architecture, caching strategies, performance optimization

---

**Last Updated**: 2024-01-02
**Author**: [Your Name]
**Project**: Beaver ARS - AI-Powered Automatic Response System
