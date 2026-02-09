.PHONY: dev dev-backend dev-frontend build test clean install

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
	cp -r frontend/build/* src/cellquant/static/
	python -m build

# Install
install:
	pip install -e ".[all,dev]"

# Test
test:
	pytest tests/ -v

test-core:
	pytest tests/python/test_core/ -v

test-api:
	pytest tests/python/test_api/ -v

# Clean
clean:
	rm -rf dist/ build/ src/cellquant/static/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
