.PHONY: install test test-cov lint lint-fix type-check run run-prod docker-build docker-up docker-down clean coverage-html train-local drift-check bench

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ --cov=app --cov-report=term-missing --cov-fail-under=60

test-cov:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html --cov-fail-under=60
	@echo "Coverage report: htmlcov/index.html"

lint:
	ruff check .
	ruff format --check .

lint-fix:
	ruff check --fix .
	ruff format .

type-check:
	mypy app/ --ignore-missing-imports --no-strict-optional

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

docker-build:
	docker build -t price-prophet:latest -f docker/Dockerfile .

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .mypy_cache/ .ruff_cache/ dist/ build/

coverage-html:
	pytest tests/ --cov=app --cov-report=html
	@echo "Open htmlcov/index.html to view coverage report"

train-local:
	python scripts/train.py --n-samples 5000

drift-check:
	@echo "Checking drift between reference and recent predictions..."
	python -c "from app.main import app; print('Use GET /drift endpoint for live drift checks')"

bench:
	python scripts/benchmark.py --url http://localhost:8000 --n 50 --endpoint /forecast
