# Makefile for the Solar-to-Laser System

.PHONY: help clean test lint security performance integration docs build install run docker-build docker-run k8s-deploy

# Default target
help:
	@echo "Available targets:"
	@echo "  help        - Show this help message"
	@echo "  clean       - Remove build artifacts and temporary files"
	@echo "  test        - Run unit tests"
	@echo "  lint        - Run linting checks"
	@echo "  security    - Run security checks"
	@echo "  performance - Run performance tests"
	@echo "  integration - Run integration tests"
	@echo "  docs        - Build documentation"
	@echo "  build       - Build the package"
	@echo "  install     - Install the package"
	@echo "  run         - Run the system"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  k8s-deploy   - Deploy to Kubernetes"

# Clean build artifacts and temporary files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .tox/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Run unit tests
test:
	pytest tests/ --cov=src --cov-report=term --cov-report=xml

# Run linting checks
lint:
	black --check src tests
	isort --check-only src tests
	mypy src tests
	flake8 src tests
	pylint src tests

# Run security checks
security:
	bandit -r src
	safety check
	pytest tests/test_security.py

# Run performance tests
performance:
	pytest tests/test_performance.py --benchmark-autosave --benchmark-compare

# Run integration tests
integration:
	pytest tests/test_integration.py

# Build documentation
docs:
	sphinx-build -b html docs/source docs/build/html

# Build the package
build:
	python setup.py build

# Install the package
install:
	pip install -e .

# Run the system
run:
	python -m src.main all

# Build Docker images
docker-build:
	python -m src.main docker-build

# Run with Docker Compose
docker-run:
	python -m src.main compose --up

# Deploy to Kubernetes
k8s-deploy:
	python -m src.main k8s

# Format code
format:
	black src tests
	isort src tests

# Run all tests
test-all:
	tox

# Create a new release
release:
	python setup.py sdist bdist_wheel
	twine upload dist/*

# Install development dependencies
dev-install:
	pip install -e ".[dev]"

# Run a specific module
run-data:
	python -m src.main data

run-audio:
	python -m src.main audio

run-rave:
	python -m src.main rave

run-vector:
	python -m src.main vector

run-laser:
	python -m src.main laser

run-deploy:
	python -m src.main deploy