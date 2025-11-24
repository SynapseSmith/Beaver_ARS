#!/bin/bash

# ================================
# Beaver ARS Rollback Script
# ================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="beaver-ars"
BACKUP_DIR="/opt/backups"

echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}  Beaver ARS Rollback Script${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

# List available backups
list_backups() {
    echo -e "\n${YELLOW}Available backups:${NC}"
    
    if [ ! -d "${BACKUP_DIR}" ] || [ -z "$(ls -A ${BACKUP_DIR}/${PROJECT_NAME}_*.tar.gz 2>/dev/null)" ]; then
        echo -e "${RED}No backups found in ${BACKUP_DIR}${NC}"
        exit 1
    fi
    
    cd ${BACKUP_DIR}
    ls -lh ${PROJECT_NAME}_*.tar.gz | awk '{print NR". "$9" ("$5")"}'
}

# Select backup
select_backup() {
    list_backups
    
    echo -e "\n${YELLOW}Select backup number to restore (or 'q' to quit):${NC}"
    read -p "> " selection
    
    if [ "$selection" = "q" ]; then
        echo "Rollback cancelled"
        exit 0
    fi
    
    BACKUP_FILE=$(ls -t ${BACKUP_DIR}/${PROJECT_NAME}_*.tar.gz | sed -n "${selection}p")
    
    if [ -z "$BACKUP_FILE" ]; then
        echo -e "${RED}Invalid selection${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Selected: ${BACKUP_FILE}${NC}"
}

# Confirm rollback
confirm_rollback() {
    echo -e "\n${RED}WARNING: This will restore the application to a previous state.${NC}"
    echo -e "${RED}Current data will be overwritten.${NC}"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Rollback cancelled"
        exit 0
    fi
}

# Stop services
stop_services() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    
    cd /opt/${PROJECT_NAME}
    docker-compose down
    
    echo "  ✓ Services stopped"
}

# Restore backup
restore_backup() {
    echo -e "\n${YELLOW}Restoring backup...${NC}"
    
    # Create temporary backup of current state
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TEMP_BACKUP="/tmp/${PROJECT_NAME}_pre_rollback_${TIMESTAMP}.tar.gz"
    tar -czf ${TEMP_BACKUP} -C /opt ${PROJECT_NAME}
    echo "  ✓ Current state backed up to ${TEMP_BACKUP}"
    
    # Remove current installation
    rm -rf /opt/${PROJECT_NAME}
    
    # Extract backup
    tar -xzf ${BACKUP_FILE} -C /opt/
    
    echo "  ✓ Backup restored"
}

# Start services
start_services() {
    echo -e "\n${YELLOW}Starting services...${NC}"
    
    cd /opt/${PROJECT_NAME}
    docker-compose up -d
    
    echo "  Waiting for services to be ready..."
    sleep 10
    
    echo "  ✓ Services started"
}

# Health check
health_check() {
    echo -e "\n${YELLOW}Running health check...${NC}"
    
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

# Main rollback flow
main() {
    select_backup
    confirm_rollback
    stop_services
    restore_backup
    start_services
    
    if health_check; then
        echo -e "\n${GREEN}✓ Rollback completed successfully!${NC}\n"
        exit 0
    else
        echo -e "\n${RED}✗ Rollback failed! Services may be in an inconsistent state.${NC}"
        echo -e "${YELLOW}Check logs with: docker-compose logs${NC}\n"
        exit 1
    fi
}

# Run rollback
main
