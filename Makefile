.PHONY: help setup up down logs ps bootstrap train test lint clean demo-drift

help:
	@echo "setup        Copy .env.example -> .env (edit the passwords afterwards)"
	@echo "up           Build and start the whole stack"
	@echo "down         Stop the stack (keeps volumes)"
	@echo "bootstrap    Create the schema and seed stations + historical departures"
	@echo "train        Train candidates and run champion/challenger promotion"
	@echo "test         Run the test suite"
	@echo "lint         Run ruff"
	@echo "demo-drift   Induce drift and show the monitoring verdict"
	@echo "clean        Stop the stack and delete volumes (destroys all data)"

setup:
	@test -f .env || cp .env.example .env
	@echo ".env ready - edit the passwords before exposing this anywhere"

up:
	docker compose up -d --build
	@echo "API       http://localhost:8000/docs"
	@echo "MLflow    http://localhost:5001"
	@echo "Airflow   http://localhost:8080"
	@echo "Dashboard http://localhost:8501"

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f --tail=100

bootstrap:
	DATABASE_URL=postgresql+psycopg2://citibike:citibike@localhost:5432/citibike \
		python -m scripts.bootstrap_db

train:
	MLFLOW_TRACKING_URI=http://localhost:5001 \
	DATABASE_URL=postgresql+psycopg2://citibike:citibike@localhost:5432/citibike \
		python -m src.models.train --source offline

test:
	pytest -q

lint:
	ruff check src tests dags scripts

demo-drift:
	MLFLOW_TRACKING_URI=http://localhost:5001 \
	DATABASE_URL=postgresql+psycopg2://citibike:citibike@localhost:5432/citibike \
		python -m scripts.simulate_drift --baseline --inject --check

clean:
	docker compose down -v
