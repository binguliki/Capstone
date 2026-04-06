.PHONY: help install dev test run docker-build docker-up docker-down deploy-local deploy-docker clean lint format

help:
	@echo "Leaf Disease Detection API - Makefile Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install dependencies"
	@echo "  make dev-setup      - Setup development environment"
	@echo ""
	@echo "Development:"
	@echo "  make dev            - Run server with auto-reload"
	@echo "  make test           - Run test suite"
	@echo "  make test-smoke     - Run smoke tests only"
	@echo "  make test-core      - Run core functionality tests"
	@echo ""
	@echo "Deployment:"
	@echo "  make run            - Run server (production mode)"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-up      - Start with Docker Compose"
	@echo "  make docker-down    - Stop Docker Compose"
	@echo "  make docker-logs    - View Docker logs"
	@echo ""
	@echo "Production:"
	@echo "  make prod           - Run with Gunicorn (4 workers)"
	@echo "  make prod-scale     - Run with auto-scaled workers"
	@echo ""
	@echo "Utilities:"
	@echo "  make lint           - Lint code with pylint"
	@echo "  make format         - Format code with black"
	@echo "  make clean          - Clean temporary files"
	@echo "  make health         - Check API health"
	@echo "  make predict-test   - Test prediction endpoint"
	@echo ""

# Setup & Installation

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

dev-setup: install
	pip install pytest pytest-asyncio gunicorn
	cp .env.example .env
	@echo "✓ Development environment ready"

# Development

dev:
	@echo "Starting development server with auto-reload..."
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest test_api.py -v --tb=short

test-smoke:
	pytest test_api.py -v -m smoke --tb=short

test-core:
	pytest test_api.py -v -m core --tb=short

test-performance:
	pytest test_api.py -v -m performance --tb=short

# Regular execution

run:
	python main.py

# Docker

docker-build:
	docker build -t leaf-disease-api:latest .
	@echo "✓ Docker image built: leaf-disease-api:latest"

docker-up:
	docker-compose up --build
	@echo "✓ Started with Docker Compose"

docker-down:
	docker-compose down
	@echo "✓ Stopped Docker Compose"

docker-logs:
	docker-compose logs -f api

docker-clean:
	docker-compose down -v
	docker system prune -f

# Production

prod:
	gunicorn -c gunicorn_config.py main:app --workers 4

prod-scale:
	WORKERS=$$(($$(nproc) * 2 + 1)) gunicorn -c gunicorn_config.py main:app

# Utilities

lint:
	pylint main.py client.py --disable=too-many-arguments,too-many-locals,missing-docstring
	@echo "✓ Linting complete"

format:
	black main.py client.py test_api.py
	@echo "✓ Formatting complete"

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned temporary files"

# API Testing

health:
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo ""

metrics:
	@curl -s http://localhost:8000/metrics | python -m json.tool
	@echo ""

predict-test:
	@echo "Creating test image..."
	@python -c "from test_api import TestAPIEndpoints; import io; img = TestAPIEndpoints.create_test_image(); print('Test image ready')"
	@echo "Making prediction..."
	@curl -X POST http://localhost:8000/predict -F "file=@test.jpg" -s | python -m json.tool
	@echo ""

# Database (if applicable)

db-migrate:
	@echo "Running migrations..."
	@echo "Not configured yet"

# Documentation

docs:
	@echo "Documentation files:"
	@echo "  - README.md      : Main documentation"
	@echo "  - DEPLOYMENT.md  : Deployment guide"
	@echo "  - ARCHITECTURE.md: Architecture overview"

# Info

info:
	@echo "System Information:"
	@python -c "import sys; print(f'Python: {sys.version}')"
	@echo "Project Structure:"
	@find . -type f -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "Dockerfile" | grep -v venv | sort
