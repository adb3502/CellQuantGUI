.PHONY: dev dev-backend dev-frontend build test clean install docker

# Development
dev-backend:
	uvicorn cellquant.api.app:create_app --factory --reload --host 127.0.0.1 --port 8000

dev-frontend:
	cd frontend && npm run dev -- --port 5173

dev:
	$(MAKE) dev-backend & $(MAKE) dev-frontend

# Build
build-frontend:
	cd frontend && npm ci && npm run build

build: build-frontend
	mkdir -p src/cellquant/static
	cp -r frontend/build/* src/cellquant/static/
	python -m build

# Install
install:
	pip install -e ".[all,dev]"

install-frontend:
	cd frontend && npm ci

# Test
test:
	pytest tests/ -v

test-core:
	pytest tests/python/test_core/ -v

test-api:
	pytest tests/python/test_api/ -v

# Docker
docker:
	docker build -t cellquant:latest .

docker-run:
	docker run --rm -p 8000:8000 --gpus all cellquant:latest

# Clean
clean:
	rm -rf dist/ build/ src/cellquant/static/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Session cleanup
cleanup:
	python -m cellquant cleanup --max-age 24
