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
	docker-down docker-logs docker-ps clean \
	mlflow-up serving-up infra-down logs-api logs-dashboard simulate simulate-remote

help:
	@echo "Comandos disponíveis:"
	@echo "--- Operação Local (Bare Metal) ---"
	@echo "  make setup        - cria .venv, instala dependências e prepara diretórios"
	@echo "  make train        - executa pipeline de treinamento (modo multi-ano)"
	@echo "  make api          - sobe API FastAPI local (reload)"
	@echo "  make dashboard    - sobe dashboard Streamlit local"
	@echo "  make simulate     - gera tráfego, drift e feedback para popular dashboards locais"
	@echo "  make simulate-remote URL=http://seu-k8s-ip - gera massa de dados para o ambiente remoto"
	@echo "--- Operação Docker Compose (Stack Completa) ---"
	@echo "  make mlflow-up    - sobe apenas o servidor MLflow (detached)"
	@echo "  make serving-up   - sobe API e Dashboard conectados ao MLflow"
	@echo "  make docker-up    - sobe toda a stack (MLflow + API + Dashboard)"
	@echo "  make docker-down  - derruba toda a stack e limpa redes"
	@echo "--- Qualidade e Testes ---"
	@echo "  make test         - executa suíte de testes"
	@echo "  make test-cov     - executa testes com cobertura"
	@echo "  make lint         - valida estilo e tipos (mypy com config)"
	@echo "  make format       - formata código com black/isort"
	@echo "--- utilitários ---"
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
	$(PYTHON) scripts/train_model.py --year all

api: setup
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

dashboard: setup
	$(STREAMLIT) run dashboard/app.py --server.port 8501

# --- Docker Orchestration ---

mlflow-up: dirs
	$(DOCKER_COMPOSE) up -d mlflow

serving-up: dirs
	$(DOCKER_COMPOSE) up -d --build api dashboard

infra-down:
	$(DOCKER_COMPOSE) down

docker-up: dirs
	$(DOCKER_COMPOSE) up --build

docker-up-d: dirs
	$(DOCKER_COMPOSE) up -d --build

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

logs-api:
	$(DOCKER_COMPOSE) logs -f api

logs-dashboard:
	$(DOCKER_COMPOSE) logs -f dashboard

docker-ps:
	$(DOCKER_COMPOSE) ps

# --- Tests and Quality ---

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
	$(VENV)/bin/black --check src api tests scripts
	$(VENV)/bin/isort --check-only src api tests scripts
	$(VENV)/bin/flake8 src api tests scripts
	$(VENV)/bin/mypy src api scripts --config-file pyproject.toml

format: setup
	$(VENV)/bin/black src api tests scripts
	$(VENV)/bin/isort src api tests scripts

simulate: setup
	$(PYTHON) scripts/simulate_production.py

simulate-remote: setup
	@if [ -z "$(URL)" ]; then \
		echo "Erro: URL não definida. Use 'make simulate-remote URL=http://seu-ip-k8s'"; \
		exit 1; \
	fi
	API_URL=$(URL) $(PYTHON) scripts/simulate_production.py

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
