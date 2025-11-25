#!/bin/bash

# ================================
# Beaver ARS Deployment Script
# ================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Beaver_ARS"
DOCKER_IMAGE="beaver-ars:latest"
BACKUP_DIR="/opt/backups"
MAX_BACKUPS=5

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  Beaver ARS Deployment Script${NC}"
echo -e "${GREEN}=====================================${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

# Function: Check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}[1/8] Checking prerequisites...${NC}"
    
    commands=("docker" "docker-compose" "git")
    for cmd in "${commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${RED}Error: $cmd is not installed${NC}"
            exit 1
        fi
        echo "  ✓ $cmd found"
    done
}

# Function: Pull latest code
pull_code() {
    echo -e "\n${YELLOW}[2/8] Pulling latest code...${NC}"
    
    cd /opt/fastapi-poc/${PROJECT_NAME}
    git fetch origin
    git pull origin main
    echo "  ✓ Code updated"
}

# Function: Backup current deployment
backup_deployment() {
    echo -e "\n${YELLOW}[3/8] Creating backup...${NC}"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="${BACKUP_DIR}/${PROJECT_NAME}_${TIMESTAMP}.tar.gz"
    
    mkdir -p ${BACKUP_DIR}
    
    tar -czf ${BACKUP_FILE} \
        --exclude='logs/*' \
        --exclude='*.log' \
        --exclude='wandb/*' \
        --exclude='models/*' \
        --exclude='.git/*' \
        -C /opt/fastapi-poc ${PROJECT_NAME}
    
    echo "  ✓ Backup created: ${BACKUP_FILE}"
    
    # Keep only last N backups
    cd ${BACKUP_DIR}
    ls -t ${PROJECT_NAME}_*.tar.gz | tail -n +$((MAX_BACKUPS + 1)) | xargs -r rm
}

# Function: Update environment variables
update_env() {
    echo -e "\n${YELLOW}[4/8] Updating environment variables...${NC}"
    
    cd /opt/fastapi-poc/${PROJECT_NAME}
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            echo -e "  ${YELLOW}Warning: .env file created from .env.example${NC}"
            echo -e "  ${YELLOW}Please update it with production values${NC}"
        else
            echo -e "  ${YELLOW}Warning: No .env or .env.example found${NC}"
        fi
    else
        echo "  ✓ .env file exists"
    fi
}

# Function: Build Docker image
build_image() {
    echo -e "\n${YELLOW}[5/8] Building Docker image...${NC}"
    
    cd /opt/fastapi-poc/${PROJECT_NAME}
    docker-compose build --no-cache
    echo "  ✓ Image built successfully"
}

# Function: Run database migrations
run_migrations() {
    echo -e "\n${YELLOW}[6/8] Running database migrations...${NC}"
    
    cd /opt/fastapi-poc/${PROJECT_NAME}
    
    # Check if MySQL container is running
    if docker-compose ps mysql | grep -q "Up"; then
        docker-compose exec -T mysql mysql -uroot -p${MYSQL_ROOT_PASSWORD:-root_password} ${MYSQL_DATABASE:-beaver_ars} < database/init.sql 2>/dev/null || true
        echo "  ✓ Migrations completed"
    else
        echo "  ⊘ MySQL not running, skipping migrations"
    fi
}

# Function: Start services
start_services() {
    echo -e "\n${YELLOW}[7/8] Starting services...${NC}"
    
    cd /opt/fastapi-poc/${PROJECT_NAME}
    docker-compose down
    docker-compose up -d
    
    echo "  Waiting for services to be ready..."
    sleep 30
    
    echo "  ✓ Services started"
}

# Function: Health check
health_check() {
    echo -e "\n${YELLOW}[8/8] Running health check...${NC}"
    
    MAX_RETRIES=5
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -f http://localhost:5000/health &> /dev/null; then
            echo -e "  ${GREEN}✓ Health check passed${NC}"
            return 0
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "  Attempt $RETRY_COUNT/$MAX_RETRIES failed, retrying..."
        sleep 5
    done
    
    echo -e "  ${RED}✗ Health check failed${NC}"
    return 1
}

# Function: Cleanup old Docker resources
cleanup() {
    echo -e "\n${YELLOW}Cleaning up old Docker resources...${NC}"
    
    docker system prune -f
    docker volume prune -f
    echo "  ✓ Cleanup completed"
}

# Function: Show service status
show_status() {
    echo -e "\n${GREEN}=====================================${NC}"
    echo -e "${GREEN}  Deployment Status${NC}"
    echo -e "${GREEN}=====================================${NC}"
    
    cd /opt/fastapi-poc/${PROJECT_NAME}
    docker-compose ps
    
    echo -e "\n${GREEN}Services are accessible at:${NC}"
    echo "  • Application (Direct): http://localhost:5000"
    echo "  • Nginx (Proxy): http://localhost:9080"
    echo "  • Grafana: http://localhost:3000"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • MySQL: localhost:3306"
    echo "  • Redis: localhost:6379"
}

# Main deployment flow
main() {
    check_prerequisites
    pull_code
    backup_deployment
    update_env
    build_image
    run_migrations
    start_services
    
    if health_check; then
        cleanup
        show_status
        echo -e "\n${GREEN}✓ Deployment completed successfully!${NC}\n"
        exit 0
    else
        echo -e "\n${RED}✗ Deployment failed!${NC}"
        echo -e "${YELLOW}Rolling back...${NC}"
        ./rollback.sh
        exit 1
    fi
}

# Run deployment
main
