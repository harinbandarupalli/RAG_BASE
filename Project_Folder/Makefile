.PHONY: help setup docker-up docker-down run-api run-ui test lint format clean

help:
	@echo "Available commands:"
	@echo "  make setup        Install project dependencies"
	@echo "  make docker-up    Start Postgres (pgvector) and Redis"
	@echo "  make docker-down  Stop all running containers"
	@echo "  make run-api      Start FastAPI backend on port 8000"
	@echo "  make run-ui       Start Streamlit decoupled UI on port 8501"
	@echo "  make test         Run test suite with pytest"
	@echo "  make lint         Run ruff and mypy checks"
	@echo "  make format       Format code using black and ruff"

setup:
	pip install -e ".[dev]"

docker-up:
	docker compose up -d postgres redis

docker-down:
	docker compose down

docker-phoenix:
	docker compose --profile observability up -d

db-migrate:
	alembic upgrade head

db-revision:
	alembic revision --autogenerate -m "$(msg)"

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run ui/app.py --server.port 8501

test:
	pytest tests/ -v

lint:
	ruff check src/
	mypy src/

format:
	ruff check --fix src/
	black src/ ui/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
