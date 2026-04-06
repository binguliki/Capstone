#!/bin/bash
# Quick start script for Leaf Disease Detection API

set -e

echo "🌿 Leaf Disease Detection API - Quick Start"
echo "==========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}[1/5] Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create virtual environment
echo -e "${BLUE}[2/5] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${BLUE}[3/5] Installing dependencies...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check model file
echo -e "${BLUE}[4/5] Checking model file...${NC}"
if [ ! -f "modelv1.keras" ]; then
    echo -e "${RED}⚠️  modelv1.keras not found in current directory${NC}"
    echo "Please ensure modelv1.keras is in the model-backend directory"
else
    FILE_SIZE=$(ls -lh modelv1.keras | awk '{print $5}')
    echo -e "${GREEN}✓ Model found (size: $FILE_SIZE)${NC}"
fi

# Setup environment
echo -e "${BLUE}[5/5] Setting up environment...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Start server
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Starting Leaf Disease Detection API..."
echo ""
echo "📊 API Documentation:"
echo "   • Swagger UI: http://localhost:8000/docs"
echo "   • ReDoc: http://localhost:8000/redoc"
echo ""
echo "🔗 Endpoints:"
echo "   • POST   /predict  - Disease detection"
echo "   • GET    /health   - Health check"
echo "   • GET    /metrics  - Performance metrics"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
