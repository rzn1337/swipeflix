.PHONY: help dev start train train-full lint fmt test docker-build docker-run compose-up compose-down precommit clean

# Colors for output
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
RESET  := $(shell tput -Txterm sgr0)

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${GREEN}%-20s${RESET} %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Create virtual environment and install dependencies
	@echo "${GREEN}Creating virtual environment...${RESET}"
	python3.11 -m venv venv
	@echo "${GREEN}Installing dependencies...${RESET}"
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install -r requirements-dev.txt
	@echo "${GREEN}Installing pre-commit hooks...${RESET}"
	./venv/bin/pre-commit install
	@echo "${GREEN}Setup complete! Activate with: source venv/bin/activate${RESET}"

start: ## Run FastAPI server locally
	@echo "${GREEN}Starting SwipeFlix API server...${RESET}"
	PYTHONPATH=./src uvicorn swipeflix.api.main:app --host 0.0.0.0 --port 8000 --reload

train: ## Train model with sample data (fast for development)
	@echo "${GREEN}Training model with sample data...${RESET}"
	PYTHONPATH=./src python src/swipeflix/ml/train.py

train-full: ## Train model on full dataset
	@echo "${GREEN}Training model on full dataset...${RESET}"
	PYTHONPATH=./src python src/swipeflix/ml/train.py

lint: ## Run linter (ruff)
	@echo "${GREEN}Running ruff linter...${RESET}"
	ruff check src/ tests/
	@echo "${GREEN}Running black check...${RESET}"
	black --check src/ tests/

fmt: ## Format code with black and ruff
	@echo "${GREEN}Formatting code with black...${RESET}"
	black src/ tests/
	@echo "${GREEN}Running ruff fixes...${RESET}"
	ruff check --fix src/ tests/

test: ## Run tests with coverage
	@echo "${GREEN}Running tests with coverage...${RESET}"
	PYTHONPATH=./src pytest tests/ --cov=src/swipeflix --cov-report=term --cov-report=html --cov-fail-under=80 -v

test-quick: ## Run tests without coverage (faster)
	@echo "${GREEN}Running quick tests...${RESET}"
	PYTHONPATH=./src pytest tests/ -v

docker-build: ## Build Docker image
	@echo "${GREEN}Building Docker image...${RESET}"
	DOCKER_BUILDKIT=1 docker build -t swipeflix:latest .

docker-build-gpu: ## Build GPU-enabled Docker image
	@echo "${GREEN}Building GPU Docker image...${RESET}"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.gpu -t swipeflix:gpu .

docker-run: ## Run Docker container locally
	@echo "${GREEN}Running Docker container...${RESET}"
	docker run -d -p 8000:8000 --name swipeflix swipeflix:latest
	@echo "${GREEN}Container started! Access at http://localhost:8000${RESET}"
	@echo "${YELLOW}Stop with: docker stop swipeflix && docker rm swipeflix${RESET}"

docker-stop: ## Stop and remove Docker container
	@echo "${GREEN}Stopping Docker container...${RESET}"
	docker stop swipeflix || true
	docker rm swipeflix || true

compose-up: ## Start Docker Compose stack (dev profile)
	@echo "${GREEN}Starting Docker Compose stack (dev profile)...${RESET}"
	docker-compose --profile dev up --build -d
	@echo "${GREEN}Services started!${RESET}"
	@echo "  - API:        http://localhost:8000"
	@echo "  - MLflow:     http://localhost:5000"
	@echo "  - MinIO:      http://localhost:9001"
	@echo "  - Grafana:    http://localhost:3000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Evidently:  http://localhost:7000"

compose-down: ## Stop Docker Compose stack
	@echo "${GREEN}Stopping Docker Compose stack...${RESET}"
	docker-compose --profile dev --profile test --profile prod down

compose-logs: ## View Docker Compose logs
	docker-compose logs -f

precommit: ## Run pre-commit hooks on all files
	@echo "${GREEN}Running pre-commit hooks...${RESET}"
	pre-commit run --all-files

precommit-update: ## Update pre-commit hook versions
	@echo "${GREEN}Updating pre-commit hooks...${RESET}"
	pre-commit autoupdate

acceptance-tests: ## Run acceptance tests against running service
	@echo "${GREEN}Running acceptance tests...${RESET}"
	bash scripts/acceptance_tests.sh

security-scan: ## Run security vulnerability scan
	@echo "${GREEN}Running pip-audit...${RESET}"
	pip-audit --desc

terraform-init: ## Initialize Terraform
	@echo "${GREEN}Initializing Terraform...${RESET}"
	cd infra/terraform && terraform init

terraform-plan: ## Plan Terraform changes
	@echo "${GREEN}Planning Terraform changes...${RESET}"
	cd infra/terraform && terraform plan

terraform-apply: ## Apply Terraform configuration
	@echo "${GREEN}Applying Terraform configuration...${RESET}"
	cd infra/terraform && terraform apply -auto-approve

terraform-destroy: ## Destroy Terraform resources
	@echo "${YELLOW}Destroying Terraform resources...${RESET}"
	cd infra/terraform && terraform destroy -auto-approve

dvc-pull: ## Pull data from DVC remote
	@echo "${GREEN}Pulling data from DVC remote...${RESET}"
	dvc pull

dvc-push: ## Push data to DVC remote
	@echo "${GREEN}Pushing data to DVC remote...${RESET}"
	dvc push

k6-test: ## Run k6 load tests
	@echo "${GREEN}Running k6 load tests...${RESET}"
	k6 run k6/script.js

clean: ## Clean up temporary files and caches
	@echo "${GREEN}Cleaning up...${RESET}"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/
	@echo "${GREEN}Cleanup complete!${RESET}"

clean-all: clean ## Clean everything including venv and Docker
	@echo "${YELLOW}Removing virtual environment...${RESET}"
	rm -rf venv/
	@echo "${YELLOW}Removing MLflow runs...${RESET}"
	rm -rf mlruns/
	rm -rf mlartifacts/
	@echo "${YELLOW}Stopping Docker containers...${RESET}"
	docker-compose down -v 2>/dev/null || true
	@echo "${GREEN}Complete cleanup done!${RESET}"
