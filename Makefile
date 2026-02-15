.PHONY: install test lint train serve docker-build docker-run clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ serve/ --max-line-length=120 --ignore=E501,W503

train:
	python -m src.loan_pipeline

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t loan-mlops .

docker-run:
	docker run -p 8000:8000 loan-mlops

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache mlruns *.egg-info
