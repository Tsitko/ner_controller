#!/bin/bash
# Setup script for supervisor autostart

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Installing supervisor...${NC}"
sudo apt-get update
sudo apt-get install -y supervisor

echo -e "${YELLOW}Creating supervisor config...${NC}"

# Get absolute path to project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create supervisor config for ner_controller application
sudo tee /etc/supervisor/conf.d/ner_controller.conf > /dev/null <<EOF
[program:ner_controller]
command=${PROJECT_DIR}/venv/bin/uvicorn ner_controller.main:app --host 0.0.0.0 --port 1304
directory=${PROJECT_DIR}/src
autostart=true
autorestart=true
startretries=3
stderr_logfile=${PROJECT_DIR}/logs/ner_controller_error.log
stdout_logfile=${PROJECT_DIR}/logs/ner_controller_output.log
user=$(whoami)
environment=PATH="${PROJECT_DIR}/venv/bin:%(ENV_PATH)s",PYTHONPATH="${PROJECT_DIR}/src",HF_HOME="/home/denis/.cache/huggingface",TRANSFORMERS_OFFLINE="1",HF_HUB_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1"
EOF

echo -e "${YELLOW}Creating logs directory...${NC}"
mkdir -p ${PROJECT_DIR}/logs

echo -e "${YELLOW}Reloading supervisor configuration...${NC}"
sudo supervisorctl reread
sudo supervisorctl update

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Useful commands:"
echo "  sudo supervisorctl start ner_controller    - Start service"
echo "  sudo supervisorctl stop ner_controller     - Stop service"
echo "  sudo supervisorctl restart ner_controller  - Restart service"
echo "  sudo supervisorctl status                  - Check status"
echo "  tail -f logs/ner_controller_output.log     - Watch service logs"
