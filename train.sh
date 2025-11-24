#!/bin/bash

# ================================
# Beaver ARS Complete Training Pipeline
# ================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  Beaver ARS Training Pipeline${NC}"
echo -e "${GREEN}=====================================${NC}"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Activate virtual environment
VENV_PATH="/opt/fastapi-poc/venv"
if [ -d "${VENV_PATH}" ]; then
    echo -e "\n${YELLOW}Activating virtual environment...${NC}"
    source "${VENV_PATH}/bin/activate"
    echo -e "  ${GREEN}✓ Virtual environment activated${NC}"
    PYTHON_CMD="python"
else
    echo -e "${YELLOW}⚠ Virtual environment not found, using system Python${NC}"
    PYTHON_CMD="/opt/fastapi-poc/venv/bin/python"
fi

# Configuration
DATA_DIR="./data"
MODEL_DIR="./models"
INTENT_DATA="${DATA_DIR}/sample/intent_sample.csv"
NER_DATA="${DATA_DIR}/sample/ner_sample.conll"
INTENT_MODEL_DIR="${MODEL_DIR}/intent_classifier"
NER_MODEL_DIR="${MODEL_DIR}/ner_model"

# Training parameters
INTENT_EPOCHS=20
NER_EPOCHS=15
BATCH_SIZE=4  # Reduced from 16 to prevent memory issues
LEARNING_RATE=2e-5
GRADIENT_ACCUMULATION=4  # Effective batch size = 4 * 4 = 16

# Function: Check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}[1/7] Checking prerequisites...${NC}"
    
    # Check Python
    echo "  ✓ Python found: $(${PYTHON_CMD} --version)"
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "  ✓ GPU available:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    else
        echo -e "  ${YELLOW}⚠ No GPU detected, training will be slow${NC}"
        read -p "  Continue with CPU? (y/n): " continue_cpu
        if [ "$continue_cpu" != "y" ]; then
            exit 0
        fi
    fi
    
    # Check required packages
    packages=("torch" "transformers" "datasets" "wandb" "pandas" "seqeval" "scikit-learn")
    for pkg in "${packages[@]}"; do
        if ${PYTHON_CMD} -c "import $pkg" 2>/dev/null; then
            echo "  ✓ $pkg installed"
        else
            echo -e "  ${RED}✗ $pkg not found${NC}"
            echo -e "  ${YELLOW}Installing dependencies...${NC}"
            ${PYTHON_CMD} -m pip install -r requirements.txt
            break
        fi
    done
}

# Function: Prepare directories
prepare_directories() {
    echo -e "\n${YELLOW}[2/7] Preparing directories...${NC}"
    
    mkdir -p ${MODEL_DIR}
    mkdir -p ${INTENT_MODEL_DIR}
    mkdir -p ${NER_MODEL_DIR}
    mkdir -p ${DATA_DIR}/processed
    mkdir -p logs
    
    echo "  ✓ Directories created"
}

# Function: Validate data
validate_data() {
    echo -e "\n${YELLOW}[3/7] Validating data...${NC}"
    
    # Check Intent data
    if [ ! -f "$INTENT_DATA" ]; then
        echo -e "  ${RED}✗ Intent data not found: ${INTENT_DATA}${NC}"
        echo -e "  ${YELLOW}Please prepare your Intent classification dataset${NC}"
        exit 1
    fi
    
    INTENT_LINES=$(wc -l < "$INTENT_DATA")
    echo "  ✓ Intent data found: ${INTENT_LINES} lines"
    
    if [ $INTENT_LINES -lt 100 ]; then
        echo -e "  ${YELLOW}⚠ Warning: Small dataset (< 100 samples)${NC}"
    fi
    
    # Check NER data
    if [ ! -f "$NER_DATA" ]; then
        echo -e "  ${RED}✗ NER data not found: ${NER_DATA}${NC}"
        echo -e "  ${YELLOW}Please prepare your NER dataset in CoNLL format${NC}"
        exit 1
    fi
    
    NER_LINES=$(wc -l < "$NER_DATA")
    echo "  ✓ NER data found: ${NER_LINES} lines"
}

# Function: Train Intent Classifier
train_intent() {
    echo -e "\n${YELLOW}[4/7] Training Intent Classifier...${NC}"
    echo -e "${BLUE}This may take 2-4 hours on GPU, 8-12 hours on CPU${NC}"
    echo -e "${YELLOW}Using batch size: ${BATCH_SIZE} with gradient accumulation${NC}"
    
    ${PYTHON_CMD} src/241215_step1_train_cls_intent.py \
        --data_path "${INTENT_DATA}" \
        --output_dir "${INTENT_MODEL_DIR}" \
        --num_epochs ${INTENT_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --logging_dir "logs/intent" \
        2>&1 | tee logs/intent_training.log
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "  ${GREEN}✓ Intent classifier training completed${NC}"
    else
        echo -e "  ${RED}✗ Intent classifier training failed with exit code ${EXIT_CODE}${NC}"
        if grep -q "bad_alloc\|out of memory\|CUDA out of memory" logs/intent_training.log; then
            echo -e "  ${YELLOW}Memory error detected. Try reducing batch size further.${NC}"
        fi
        exit 1
    fi
}

# Function: Evaluate Intent Classifier
evaluate_intent() {
    echo -e "\n${YELLOW}[5/7] Evaluating Intent Classifier...${NC}"
    
    # Check if model exists
    if [ ! -d "${INTENT_MODEL_DIR}" ] || [ ! -f "${INTENT_MODEL_DIR}/config.json" ]; then
        echo -e "  ${YELLOW}⚠ Intent model not found, skipping evaluation${NC}"
        return 0
    fi
    
    ${PYTHON_CMD} src/241215_step1_evaluation_cls_intent.py \
        --model_path "${INTENT_MODEL_DIR}" \
        --test_data "${INTENT_DATA}" \
        --output_xlsx "logs/intent_test_results.xlsx" \
        2>&1 | tee logs/intent_evaluation.log
    
    echo -e "  ${GREEN}✓ Intent classifier evaluation completed${NC}"
}

# Function: Train NER Model
train_ner() {
    echo -e "\n${YELLOW}[6/7] Training NER Model...${NC}"
    echo -e "${BLUE}This may take 1.5-3 hours on GPU, 6-10 hours on CPU${NC}"
    echo -e "${YELLOW}Using batch size: ${BATCH_SIZE} with gradient accumulation${NC}"
    
    ${PYTHON_CMD} src/241218_step1_ner_train_i_tagging.py \
        --data_path "${NER_DATA}" \
        --output_dir "${NER_MODEL_DIR}" \
        --num_epochs ${NER_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --logging_dir "logs/ner" \
        2>&1 | tee logs/ner_training.log
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "  ${GREEN}✓ NER model training completed${NC}"
    else
        echo -e "  ${RED}✗ NER model training failed with exit code ${EXIT_CODE}${NC}"
        if grep -q "bad_alloc\|out of memory\|CUDA out of memory" logs/ner_training.log; then
            echo -e "  ${YELLOW}Memory error detected. Try reducing batch size further.${NC}"
        fi
        exit 1
    fi
}

# Function: Evaluate NER Model
evaluate_ner() {
    echo -e "\n${YELLOW}[7/7] Evaluating NER Model...${NC}"
    
    # Check if model exists
    if [ ! -d "${NER_MODEL_DIR}" ] || [ ! -f "${NER_MODEL_DIR}/config.json" ]; then
        echo -e "  ${YELLOW}⚠ NER model not found, skipping evaluation${NC}"
        return 0
    fi
    
    ${PYTHON_CMD} src/241218_step2_ner_evaluation.py \
        --model_path "${NER_MODEL_DIR}" \
        --test_data "${NER_DATA}" \
        --output_xlsx "logs/ner_test_results.xlsx" \
        2>&1 | tee logs/ner_evaluation.log
    
    echo -e "  ${GREEN}✓ NER model evaluation completed${NC}"
}

# Function: Test inference
test_inference() {
    echo -e "\n${YELLOW}Testing inference...${NC}"
    
    # Check if model exists
    if [ ! -d "${INTENT_MODEL_DIR}" ] || [ ! -f "${INTENT_MODEL_DIR}/config.json" ]; then
        echo -e "  ${YELLOW}⚠ Intent model not found, skipping inference test${NC}"
        return 0
    fi
    
    echo -e "\n${BLUE}Intent Classification Test:${NC}"
    ${PYTHON_CMD} src/241215_step1_inference_cls_intent.py \
        --model_path "${INTENT_MODEL_DIR}" \
        --text "김치찌개 2개 주문할게요"
}

# Function: Print summary
print_summary() {
    echo -e "\n${GREEN}=====================================${NC}"
    echo -e "${GREEN}  Training Summary${NC}"
    echo -e "${GREEN}=====================================${NC}"
    
    echo -e "\n${BLUE}Models saved to:${NC}"
    echo "  • Intent Classifier: ${INTENT_MODEL_DIR}/"
    echo "  • NER Model: ${NER_MODEL_DIR}/"
    
    echo -e "\n${BLUE}Logs saved to:${NC}"
    echo "  • Intent Training: logs/intent_training.log"
    echo "  • Intent Evaluation: logs/intent_evaluation.log"
    echo "  • NER Training: logs/ner_training.log"
    echo "  • NER Evaluation: logs/ner_evaluation.log"
    
    echo -e "\n${BLUE}Next steps:${NC}"
    echo "  1. Check evaluation metrics in logs/"
    echo "  2. Test the system: python src/main_system.py"
    echo "  3. Start API server: python src/web_server.py"
    echo "  4. Deploy with Docker: docker-compose up -d"
    
    echo -e "\n${GREEN}✓ Training pipeline completed successfully!${NC}\n"
}

# Main training flow
main() {
    START_TIME=$(date +%s)
    
    check_prerequisites
    prepare_directories
    validate_data
    
    # Ask for confirmation
    echo -e "\n${YELLOW}Ready to start training?${NC}"
    echo "  • Intent Classifier: ${INTENT_EPOCHS} epochs"
    echo "  • NER Model: ${NER_EPOCHS} epochs"
    echo "  • Batch Size: ${BATCH_SIZE}"
    echo "  • Learning Rate: ${LEARNING_RATE}"
    read -p "Continue? (y/n): " confirm
    
    if [ "$confirm" != "y" ]; then
        echo "Training cancelled"
        exit 0
    fi
    
    train_intent
    evaluate_intent
    train_ner
    evaluate_ner
    test_inference
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    
    echo -e "\n${BLUE}Total training time: ${HOURS}h ${MINUTES}m${NC}"
    
    print_summary
}

# Run training pipeline
main
