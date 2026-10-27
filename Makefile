# Convenience wrappers. Everything here is a documented docker compose command.
.PHONY: help up down logs shell migrate revision ingest test lint typecheck fmt check clean

COMPOSE ?= docker compose
SEASON  ?= E2025

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Start the stack and apply migrations
	$(COMPOSE) up -d --build
	$(COMPOSE) exec -T backend alembic -c backend/alembic.ini upgrade head

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Tail backend logs
	$(COMPOSE) logs -f backend

shell: ## Shell into the backend container
	$(COMPOSE) exec backend bash

migrate: ## Apply migrations
	$(COMPOSE) exec -T backend alembic -c backend/alembic.ini upgrade head

revision: ## Autogenerate a migration: make revision M="add x"
	$(COMPOSE) exec -T backend alembic -c backend/alembic.ini revision --autogenerate -m "$(M)"

ingest: ## Ingest a season: make ingest SEASON=E2025
	$(COMPOSE) exec -T backend euroleague-ai ingest all --season $(SEASON)

test: ## Run the full test suite
	$(COMPOSE) exec -T backend pytest

lint: ## Ruff lint and format check
	$(COMPOSE) exec -T backend ruff check backend
	$(COMPOSE) exec -T backend ruff format --check backend

typecheck: ## mypy
	$(COMPOSE) exec -T backend mypy

fmt: ## Ruff format + autofix
	$(COMPOSE) exec -T backend ruff format backend
	$(COMPOSE) exec -T backend ruff check --fix backend

check: lint typecheck test ## Everything CI would run

clean: ## Stop the stack and delete volumes
	$(COMPOSE) down -v
