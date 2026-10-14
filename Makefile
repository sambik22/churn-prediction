.PHONY: install data train test lint api notebooks docker-build docker-run all

install:
	pip install -r requirements-dev.txt && pip install -e .

data:
	python -c "from churn.data import download_data; print(download_data())"

train:
	python -m churn.train

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

api:
	uvicorn api.main:app --reload

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

docker-build:
	docker build -t churn-api .

docker-run:
	docker run --rm -p 8000:8000 churn-api

all: install train test
