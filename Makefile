# Copyright (c) 2026 Luca and Pam. All rights reserved.

.PHONY: help up stop down restart status logs logs-follow start-api clean

COMPOSE_FILE := docker-compose.yml
SERVE_API_SCRIPT := scripts/serve_api.py
DEFAULT_MODEL_PATH := models/iris_model.pkl
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
API_MODEL_PATH ?= $(DEFAULT_MODEL_PATH)

up:
	docker compose -f $(COMPOSE_FILE) up -d

stop:
	docker compose -f $(COMPOSE_FILE) stop

down:
	docker compose -f $(COMPOSE_FILE) down

restart: stop up

status:
	docker compose -f $(COMPOSE_FILE) ps

logs:
	docker compose -f $(COMPOSE_FILE) logs --tail 100

logs-follow:
	docker compose -f $(COMPOSE_FILE) logs -f

start-api:
	python $(SERVE_API_SCRIPT) --host $(API_HOST) --port $(API_PORT) --model-path $(API_MODEL_PATH)

clean:
	docker compose -f $(COMPOSE_FILE) down -v

write-up:
	pandoc ./WRITEUP.md -o hw3.pdf
