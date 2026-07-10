.PHONY: help install install-dev lint format test test-cov clean run-cli run-web run-api pre-commit-install

help:
	@echo "AI Furniture Detector - Development Utilities"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters (ruff, mypy)"
	@echo "  make format           Format code (black, isort)"
	@echo "  make test             Run tests with coverage"
	@echo "  make test-cov         Show coverage report"
	@echo ""
	@echo "Running:"
	@echo "  make run-cli          Run CLI with sample image"
	@echo "  make run-web          Run Gradio web UI"
	@echo "  make run-api          Run FastAPI server"
	@echo ""
	@echo "Maintenance:"
	@echo "  make pre-commit-install  Install pre-commit hooks"
	@echo "  make clean            Clean build artifacts and cache"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check ai_furniture_detector tests
	mypy ai_furniture_detector

format:
	black ai_furniture_detector tests
	isort ai_furniture_detector tests

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=ai_furniture_detector --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '.mypy_cache' -delete
	find . -type d -name 'htmlcov' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -delete
	rm -rf build dist *.egg-info

run-cli:
	ai-furniture-detector --source ai-furniture-detector/room2.jpg --verbose

run-web:
	ai-furniture-detector-web

run-api:
	ai-furniture-detector-api

pre-commit-install:
	pre-commit install
	pre-commit run --all-files

update-deps:
	pip install --upgrade -e ".[dev]"
