SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
UVICORN := $(PYTHON) -m uvicorn
STREAMLIT := $(VENV)/bin/streamlit
DOCKER_COMPOSE ?= docker compose

.PHONY: help setup venv install env dirs train api dashboard \
	test test-cov test-file lint format docker-up docker-up-d \
	docker-down docker-logs docker-ps clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make setup        - cria .venv, instala dependências e prepara diretórios"
	@echo "  make train        - executa pipeline de treinamento local"
	@echo "  make api          - sobe API FastAPI local (reload)"
	@echo "  make dashboard    - sobe dashboard Streamlit local"
	@echo "  make test         - executa suíte de testes"
	@echo "  make test-cov     - executa testes com cobertura"
	@echo "  make test-file FILE=tests/test_api.py - roda um arquivo de teste específico"
	@echo "  make lint         - valida estilo (black/isort/flake8/mypy)"
	@echo "  make format       - formata código com black/isort"
	@echo "  make docker-up    - sobe stack Docker Compose (foreground)"
	@echo "  make docker-up-d  - sobe stack Docker Compose (detached)"
	@echo "  make docker-down  - derruba stack Docker Compose"
	@echo "  make docker-logs  - acompanha logs da stack Docker Compose"
	@echo "  make docker-ps    - lista serviços da stack Docker Compose"
	@echo "  make clean        - remove caches e artefatos temporários"

$(PYTHON):
	python3 -m venv $(VENV)

venv: $(PYTHON)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

env:
	@if [ ! -f .env ] && [ -f .env.example ]; then \
		cp .env.example .env; \
		echo ".env criado a partir de .env.example"; \
	fi

dirs:
	mkdir -p models logs feature_store/online mlruns mlartifacts

setup: install env dirs

train: setup
	$(PYTHON) scripts/train_model.py

api: setup
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

dashboard: setup
	$(STREAMLIT) run dashboard/app.py --server.port 8501

test: setup
	$(PYTEST) tests/ -v

test-cov: setup
	$(PYTEST) tests/ --cov=src --cov=api --cov-report=term-missing --cov-report=html

test-file: setup
	@if [ -z "$(FILE)" ]; then \
		echo "Uso: make test-file FILE=tests/test_api.py"; \
		exit 1; \
	fi
	$(PYTEST) $(FILE) -v

lint: setup
	$(VENV)/bin/black --check src api tests
	$(VENV)/bin/isort --check-only src api tests
	$(VENV)/bin/flake8 src api tests
	$(VENV)/bin/mypy src --ignore-missing-imports --no-error-summary

format: setup
	$(VENV)/bin/black src api tests
	$(VENV)/bin/isort src api tests

docker-up:
	$(DOCKER_COMPOSE) up --build

docker-up-d:
	$(DOCKER_COMPOSE) up -d --build

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-ps:
	$(DOCKER_COMPOSE) ps

clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
