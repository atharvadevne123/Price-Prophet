.PHONY: install test lint run docker-build clean

install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio httpx pytest-cov ruff mypy

test:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

lint:
	ruff check app/ tests/ dags/ --select E,F,W,I --ignore E501

lint-fix:
	ruff check app/ tests/ dags/ --select E,F,W,I --ignore E501 --fix

type-check:
	mypy app/ --ignore-missing-imports

run:
	uvicorn app.main:app --reload --port 8000

run-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -f model.joblib metrics.json reference_stats.json faiss_index.pkl \
	      test_price_prophet.db price_prophet.db coverage.xml .coverage

coverage-html:
	pytest tests/ -v --cov=app --cov-report=html
	@echo "Open htmlcov/index.html in your browser"

train-local:
	python -c "from app.features import generate_synthetic_training_data; from app.model import train_model; X,y=generate_synthetic_training_data(2000); m=train_model(X,y); print('RMSE:', round(m['rmse_mean'],4))"
