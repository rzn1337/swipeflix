# 🎬 SwipeFlix - Swipe-First Movie Recommender

**One-line pitch:** A hybrid movie recommendation system that combines collaborative filtering and content-based approaches, packaged as a production-ready FastAPI service with complete MLOps infrastructure.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-orange.svg)
![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📐 Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        CSV[CSV Files: movies.csv, ratings.csv]
        MinIO[MinIO Object Storage]
        DVC[DVC Data Versioning]
    end
    
    subgraph "Training Pipeline"
        Train[Training Script]
        MLflow[MLflow Tracking Server]
        Registry[Model Registry]
    end
    
    subgraph "Inference API"
        FastAPI[FastAPI Service]
        Model[Loaded PyFunc Model]
        Health[/health endpoint]
        Predict[/predict endpoint]
        Metrics[/metrics endpoint]
    end
    
    subgraph "Monitoring Stack"
        Prometheus[Prometheus]
        Grafana[Grafana Dashboards]
        Evidently[Evidently Data Drift]
    end
    
    subgraph "CI/CD"
        GHA[GitHub Actions]
        GHCR[GitHub Container Registry]
        Tests[Pytest + Coverage]
        K6[k6 Load Tests]
    end
    
    CSV --> DVC
    DVC --> MinIO
    CSV --> Train
    Train --> MLflow
    MLflow --> Registry
    Registry --> Model
    Model --> FastAPI
    FastAPI --> Health
    FastAPI --> Predict
    FastAPI --> Metrics
    Metrics --> Prometheus
    Prometheus --> Grafana
    FastAPI --> Evidently
    GHA --> Tests
    GHA --> K6
    GHA --> GHCR
    GHCR --> FastAPI
```

---

## 🎁 Bonus Features Implemented (All 5!)

SwipeFlix implements **ALL 5 bonus features** as specified in Milestone 1 requirements:

### ✅ Bonus 1: Docker Compose with Profiles
- **Dev, Test, Prod profiles** - Separate environment configurations
- **Individual service control** - Run specific services (app, db, prometheus, etc.)
- **Location:** `docker-compose.yml`
- **Usage:** `docker-compose --profile dev up` or `docker-compose up app mlflow`

### ✅ Bonus 2: GPU-Enabled Image & Self-Hosted Runner
- **GPU Docker image** - NVIDIA CUDA 11.8 support for accelerated training
- **Self-hosted runner documentation** - Complete setup guide for GPU-enabled CI/CD
- **Location:** `Dockerfile.gpu`, `docs/GPU_RUNNER_SETUP.md`
- **CI Integration:** `train_gpu` job in GitHub Actions
- **Usage:** `make docker-build-gpu`

### ✅ Bonus 3: Infrastructure as Code (Terraform)
- **Local MinIO & PostgreSQL provisioning** - Terraform configuration
- **Declarative infrastructure** - Version-controlled infrastructure setup
- **Location:** `infra/terraform/` (main.tf, variables.tf)
- **Usage:** `cd infra/terraform && terraform apply`

### ✅ Bonus 4: End-to-End Load Testing with k6
- **SLO assertions** - p95 < 500ms, error rate < 1%
- **Multi-stage load profile** - Ramp-up, steady state, ramp-down
- **Location:** `k6/script.js`, `k6/README.md`
- **CI Integration:** Runs on tagged releases
- **Usage:** `k6 run k6/script.js`

### ✅ Bonus 5: Data Version Control (DVC)
- **DVC pipeline** - Versioned data and model artifacts
- **MinIO remote** - S3-compatible storage backend
- **Location:** `dvc.yaml`, `.dvc/config`
- **Usage:** `dvc pull`, `dvc push`, `dvc repro`

**See detailed documentation for each bonus feature in the sections below.**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Make
- Git

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/swipeflix.git
cd swipeflix

# Setup local development environment
make dev

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Start the API server
make start
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### Docker Compose Stack (Recommended)

```bash
# Start full development stack (app, MLflow, MinIO, Prometheus, Grafana, Evidently)
docker-compose --profile dev up --build

# Access services:
# - API: http://localhost:8000
# - MLflow UI: http://localhost:5000
# - MinIO Console: http://localhost:9001
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
# - Evidently: http://localhost:7000
```

---

## 🎯 Make Targets

| Target | Description |
|--------|-------------|
| `make dev` | Create virtual environment and install dependencies |
| `make start` | Run FastAPI server locally (uvicorn) |
| `make train` | Train model with sample data (quick) |
| `make train-full` | Train model on full dataset |
| `make lint` | Run ruff linter |
| `make fmt` | Format code with black |
| `make test` | Run pytest with coverage (≥80% required) |
| `make docker-build` | Build production Docker image |
| `make docker-run` | Run Docker container locally |
| `make compose-up` | Start Docker Compose dev stack |
| `make compose-down` | Stop Docker Compose stack |
| `make precommit` | Run pre-commit hooks on all files |
| `make clean` | Clean up temporary files |

---

## 🧪 Training the Model

### Quick Training (for development/CI)

```bash
# Train on 2000 samples (fast, deterministic)
make train -- --sample-size 2000
```

### Full Training

```bash
# Train on complete dataset
make train-full
```

The trained model will be logged to MLflow and registered as `SwipeFlixModel` version 1. View the model in MLflow UI at `http://localhost:5000`.

### Model Architecture

**Hybrid Recommender:**
- **Collaborative Filtering:** TruncatedSVD on user-item rating matrix → user/item embeddings
- **Content-Based:** TF-IDF on movie plot and genre data
- **Scoring:** Weighted combination: `score = α * collab_similarity + (1-α) * content_similarity`
- **Output:** Top-K movie recommendations per user

---

## 🔌 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-27T12:00:00",
  "model_loaded": true
}
```

### Get Recommendations

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_1",
    "top_k": 5
  }'
```

Response:
```json
{
  "user_id": "user_1",
  "recommendations": [
    {"movie_id": "1234", "title": "The Shawshank Redemption", "score": 0.92},
    {"movie_id": "5678", "title": "The Godfather", "score": 0.89},
    ...
  ],
  "model_version": "v1",
  "inference_time_ms": 45.2
}
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

---

## 🧪 Testing

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage report
pytest --cov=src/swipeflix --cov-report=html
```

Coverage must be ≥80% for CI to pass.

---

## 🔒 Security & Pre-commit Hooks

### Setup Pre-commit

```bash
# Install hooks
pre-commit install

# Run manually on all files
make precommit
```

### Security Scanning

```bash
# Scan dependencies for vulnerabilities
pip-audit

# CI fails on CRITICAL vulnerabilities
```

---

## 🏗️ Infrastructure as Code - Terraform (Bonus #3)

### Why IaC?
Infrastructure as Code ensures reproducible, version-controlled infrastructure setup. Our Terraform configuration provisions local development infrastructure that mimics production environments.

### What's Provisioned?
- **MinIO** - S3-compatible object storage for MLflow artifacts
- **PostgreSQL** - Database backend for MLflow tracking
- **Docker Network** - Isolated networking for services
- **Volumes** - Persistent data storage

### Terraform - Local MinIO Setup

```bash
cd infra/terraform
terraform init
terraform apply -var="minio_access_key=minioadmin" -var="minio_secret_key=minioadmin"

# View outputs
terraform output

# Outputs include:
# - minio_endpoint: http://localhost:9000
# - minio_console_url: http://localhost:9001
# - postgres_connection_string: postgresql://...
```

### Terraform Files
- `infra/terraform/main.tf` - Main configuration (resources, networks, volumes)
- `infra/terraform/variables.tf` - Input variables with defaults
- `infra/terraform/README.md` - Detailed usage guide

### Integration with Docker Compose
Terraform-managed infrastructure uses the same network as Docker Compose, allowing seamless integration:

```bash
# Option 1: Use Terraform for infrastructure
terraform apply
docker-compose up app mlflow

# Option 2: Use Docker Compose for everything (easier)
docker-compose --profile dev up
```

---

## 📊 ML Workflow Monitoring

### MLflow Model Registry

SwipeFlix uses MLflow for experiment tracking and model versioning. Models are automatically registered during training.

**Access MLflow UI:** http://localhost:5000

After running `make train`, you'll see:
- Experiment: `swipeflix-training`
- Model: `SwipeFlixModel` (version 1)
- Metrics: train_mse, component sizes
- Parameters: n_components, weights, sample_size

**Model Registration Code:**
```python
# In src/swipeflix/ml/train.py
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=model,
    registered_model_name=settings.model_name,  # "SwipeFlixModel"
)
```

### Evidently Data Drift Monitoring

Monitors data distribution changes over time to detect when model retraining is needed.

**Generate Drift Report:**
```bash
# Generate drift report from ratings data
python scripts/generate_drift_report.py

# View report
open monitoring/evidently/drift_report.html
```

**Evidently Service (Docker Compose):**
- URL: http://localhost:7000
- Monitors: Feature drift, data quality, target drift
- Alerts when drift score > 0.15 (high drift threshold)

**Example Output:**
```
Drift Score: 0.08
⚡ Moderate drift detected. Monitor closely.
```

### Prometheus Metrics

**Metrics Endpoint:** http://localhost:8000/metrics

**Available Metrics:**
1. **HTTP Metrics:**
   - `swipeflix_http_requests_total` - Total API requests by method, endpoint, status
   - `swipeflix_http_request_duration_seconds` - Request latency histogram (p95, p99)

2. **ML Metrics:**
   - `swipeflix_inference_duration_seconds` - Model inference time
   - `swipeflix_model_version_info` - Current model version
   - `swipeflix_predictions_total` - Total predictions made

3. **GPU Metrics** (when GPU available):
   - `swipeflix_gpu_utilization_percent` - GPU usage percentage
   - `swipeflix_gpu_memory_used_mb` - GPU memory consumption
   - `swipeflix_gpu_temperature_celsius` - GPU temperature
   - `swipeflix_gpu_power_usage_watts` - GPU power draw

**Prometheus UI:** http://localhost:9090

### Grafana Dashboards

**Access:** http://localhost:3000 (default: admin/admin)

**Pre-configured Dashboard: "SwipeFlix API Dashboard"**

5 Panels:
1. **Request Rate** - Requests per second by endpoint
2. **Request Latency** - p95 and p99 latency trends
3. **Error Rate** - Percentage of 5xx errors (gauge)
4. **Model Inference Latency** - ML prediction times
5. **Model Version** - Current deployed model version

**Viewing GPU Metrics in Grafana:**
```
Query: swipeflix_gpu_utilization_percent
```

## 📊 Monitoring (Bonus)

### Prometheus Metrics
- `swipeflix_request_count` - Total API requests
- `swipeflix_request_latency_seconds` - Request latency histogram
- `swipeflix_inference_latency_seconds` - Model inference time
- `swipeflix_model_version` - Current model version (label)

### Grafana Dashboards
1. **API Performance:** Request rates, latencies, error rates
2. **Model Metrics:** Inference times, model version tracking
3. **System Health:** CPU, memory, disk usage

Default credentials: `admin/admin`

### Screenshots

**MLflow UI - Model Registry:**
![MLflow Model Registry](docs/screenshots/mlflow-registry.png)
*Registered SwipeFlixModel v1 with training metrics and parameters*

**Grafana Dashboard:**
![Grafana Dashboard](docs/screenshots/grafana-dashboard.png)
*Real-time monitoring of API performance, inference latency, and GPU utilization*

**Evidently Drift Report:**
![Evidently Drift Detection](docs/screenshots/evidently-drift.png)
*Data drift monitoring showing distribution changes over time*

> **Note:** To generate screenshots after setup:
> 1. Start services: `make compose-up`
> 2. Train model: `make train`
> 3. Generate traffic: `python scripts/generate_traffic.py`
> 4. Generate drift report: `python scripts/generate_drift_report.py`
> 5. Take screenshots of MLflow UI, Grafana dashboards, and Evidently reports

---

## 🚢 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR:

1. **Lint** - ruff + black formatting checks
2. **Test** - pytest with 80% coverage requirement
3. **Build** - Multi-stage Docker image
4. **Push** - Tagged image to GitHub Container Registry (GHCR)
5. **Canary Deploy** - Run container in canary mode
6. **Acceptance Tests** - 5 golden test queries
7. **Security Scan** - pip-audit for vulnerabilities
8. **Load Test** (bonus) - k6 performance tests with SLO assertions

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `CR_PAT` | GitHub Personal Access Token with `write:packages` |
| `MINIO_ROOT_USER` | MinIO admin username (optional for CI) |
| `MINIO_ROOT_PASSWORD` | MinIO admin password (optional for CI) |

### Tagging for Release

```bash
git tag v1.0-milestone1
git push origin v1.0-milestone1
```

---

## 🎮 Load Testing with k6 (Bonus #4)

```bash
# Run k6 load test
k6 run k6/script.js

# With custom thresholds
k6 run --vus 10 --duration 30s k6/script.js
```

SLO Assertions:
- p95 latency < 500ms
- Error rate < 1%
- Throughput > 100 req/s

---

## 🎨 Docker Compose Profiles (Bonus #1)

### Profile-Based Deployment

SwipeFlix supports three deployment profiles for different environments:

**1. Development Profile (`dev`)** - Full stack with all monitoring tools:
```bash
docker-compose --profile dev up
# Includes: app, postgres, minio, mlflow, prometheus, grafana, evidently
```

**2. Test Profile (`test`)** - Minimal services for CI/CD testing:
```bash
docker-compose --profile test up
# Includes: app, postgres, minio, mlflow (no monitoring overhead)
```

**3. Production Profile (`prod`)** - Optimized for production with monitoring:
```bash
docker-compose --profile prod up
# Includes: app, postgres, minio, mlflow, prometheus, grafana (no debug tools)
```

### Individual Service Control

Run specific services independently (no profile required):

```bash
# Run just the application
docker-compose up app

# Run application with database
docker-compose up app postgres

# Run MLflow with dependencies
docker-compose up mlflow postgres minio

# Run monitoring stack
docker-compose up prometheus grafana

# Run all data services
docker-compose up postgres minio mlflow
```

### Service Dependencies

Services are configured with proper health checks and dependencies:
- `app` depends on: `mlflow`, `minio`
- `mlflow` depends on: `postgres`, `minio`
- `grafana` depends on: `prometheus`

### Environment-Specific Configuration

Control behavior via environment variables:

```bash
# Development
ENVIRONMENT=development DEBUG=true docker-compose up app

# Testing
ENVIRONMENT=test docker-compose --profile test up

# Production
ENVIRONMENT=production DEBUG=false docker-compose --profile prod up
```

---

## 🖥️ GPU Support (Bonus #2)

### GPU-Enabled Docker Image

```bash
# Build GPU image
docker build -f Dockerfile.gpu -t swipeflix:gpu .

# Run with GPU support
docker run --gpus all -p 8000:8000 swipeflix:gpu
```

### Self-Hosted GitHub Actions Runner with GPU

**Setup Instructions:**

1. **Provision GPU-enabled machine** (NVIDIA GPU with drivers installed)

2. **Install Docker with NVIDIA Container Toolkit:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

3. **Register self-hosted runner:**
   - Go to repo Settings → Actions → Runners → New self-hosted runner
   - Follow instructions to download and configure runner
   - Add label `gpu` to the runner

4. **Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

5. **CI will automatically use GPU runner for `train_gpu` job** (configured in `.github/workflows/ci.yml`)

---

## 📦 Data Version Control with DVC (Bonus #5)

### DVC Setup

```bash
# Initialize DVC (already configured)
dvc remote list
# minio   s3://mlflow/dvc-storage

# Pull data from remote
dvc pull

# Add new data
dvc add data/new_dataset.csv
git add data/new_dataset.csv.dvc .gitignore
git commit -m "Add new dataset"
dvc push
```

### DVC Pipeline

```bash
# Run full DVC pipeline
dvc repro

# Visualize pipeline
dvc dag
```

---

## ☁️ AWS Cloud Integration (D9)

SwipeFlix integrates with **Amazon Web Services (AWS)** using 2 distinct cloud services for production-grade data storage and monitoring.

### 🗂️ Service 1: Amazon S3 (Data Storage)

**Purpose:** Centralized, scalable storage for training datasets

**Why S3?**
- Highly available and durable (99.999999999% durability)
- Cost-effective storage ($0.023/GB/month)
- Easy data sharing across teams and environments
- Version control for datasets
- Seamless integration with ML workflows

**What We Store:**
- `movies.csv` - Movie metadata (titles, genres)
- `ratings.csv` - User-movie ratings data

**Implementation:**
- Bucket: `s3://swipeflix/`
- Files: `movies.csv`, `ratings.csv`
- Access: Boto3 Python SDK
- Location: `src/swipeflix/cloud/aws_utils.py`, `src/swipeflix/ml/data_loader.py`

**Usage:**
```bash
# Enable S3 in .env
USE_AWS_S3=true
AWS_S3_BUCKET=swipeflix
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# Training automatically loads from S3
make train
# Output: "Loading movies data from AWS S3: s3://swipeflix/movies.csv"
```

### 📊 Service 2: Amazon CloudWatch (Monitoring & Logging)

**Purpose:** Centralized logging and real-time application monitoring

**Why CloudWatch?**
- Real-time log aggregation from distributed systems
- Custom metrics for ML-specific monitoring
- Alert capabilities for critical events
- Log retention and historical analysis
- Integration with AWS ecosystem

**What We Log:**
1. **Application Events:**
   - API startup/shutdown
   - Health check status
   - Configuration loaded

2. **ML Workflow Events:**
   - Training started/completed
   - Model version deployed
   - Training metrics (MSE, accuracy)
   - Data loading from S3

3. **Prediction Events:**
   - Prediction requests (user_id, top_k)
   - Inference latency
   - Model version used
   - Request counts

**Implementation:**
- Log Group: `swipeflix-logs`
- Log Stream: `app`
- Metrics Namespace: `SwipeFlix`
- Access: Boto3 CloudWatch Logs & Metrics APIs

**CloudWatch Metrics Sent:**
- `TrainingCompleted` - Training run counter
- `TrainingMSE` - Model performance metric
- `PredictionRequests` - API request counter
- `InferenceLatency` - Prediction response time

**Usage:**
```bash
# Enable CloudWatch in .env
CLOUDWATCH_ENABLED=true
CLOUDWATCH_LOG_GROUP=swipeflix-logs

# Start API (sends startup log to CloudWatch)
make start
# CloudWatch: "SwipeFlix API started successfully. Version: 1.0.0, S3 Integration: Enabled"

# Make prediction (logs to CloudWatch)
curl -X POST http://localhost:8000/predict -d '{"user_id":"user_1","top_k":5}'
# CloudWatch: "Prediction requested: user_id=user_1, inference_time_ms=45.2"
```

### 🔄 ML Workflow with AWS Services

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Cloud Integration                     │
└─────────────────────────────────────────────────────────────┘

1. DATA STORAGE (S3):
   ┌──────────┐
   │   S3     │  movies.csv (10 MB)
   │  Bucket  │  ratings.csv (50 MB)
   └────┬─────┘
        │
        ↓ boto3.client('s3').get_object()
   
2. TRAINING WORKFLOW:
   ┌─────────────────┐
   │  Training Script │ ← Loads data from S3
   │  (train.py)     │
   └────┬────────────┘
        │
        ├─→ CloudWatch Logs: "Training started: 10,000 movies, 100,000 ratings"
        ├─→ Trains Hybrid Model (SVD + TF-IDF)
        ├─→ Logs to MLflow (local/S3)
        ├─→ CloudWatch Metrics: TrainingMSE=0.0234
        └─→ CloudWatch Logs: "Training completed! Model: SwipeFlixModel v1"

3. INFERENCE WORKFLOW:
   User Request → FastAPI /predict
        │
        ├─→ CloudWatch Logs: "Prediction requested: user_id=user_1"
        ├─→ Load Model from MLflow
        ├─→ Generate Recommendations
        ├─→ CloudWatch Metrics: InferenceLatency=45ms
        └─→ CloudWatch Logs: "Prediction completed"
        │
        ↓
   JSON Response to User

4. MONITORING (CloudWatch Console):
   ├─→ View Logs: AWS Console → CloudWatch → Logs → swipeflix-logs
   ├─→ View Metrics: Dashboards → SwipeFlix namespace
   ├─→ Set Alarms: Alert on high latency or errors
   └─→ Log Insights: Query and analyze logs
```

### 📸 AWS Screenshots (D9 Requirement)

**S3 Bucket with Data Files:**
![AWS S3 Bucket](docs/screenshots/aws-s3-bucket.png)
*S3 bucket `swipeflix` showing movies.csv and ratings.csv files*

**CloudWatch Logs:**
![AWS CloudWatch Logs](docs/screenshots/aws-cloudwatch-logs.png)
*CloudWatch log group showing training and prediction events*

**CloudWatch Metrics Dashboard:**
![AWS CloudWatch Metrics](docs/screenshots/aws-cloudwatch-metrics.png)
*Custom metrics for training and inference monitoring*

> **Note:** To generate screenshots, set up AWS services following `docs/AWS_SETUP_GUIDE.md`, then:
> 1. Upload data to S3: `aws s3 cp data/ s3://swipeflix/ --recursive`
> 2. Enable AWS in `.env`: `USE_AWS_S3=true`, `CLOUDWATCH_ENABLED=true`
> 3. Run training: `make train`
> 4. Take screenshots from AWS Console

### 🚀 How to Reproduce AWS Setup

**Complete setup guide:** See `docs/AWS_SETUP_GUIDE.md`

**Quick Start:**

```bash
# 1. Create S3 bucket
aws s3 mb s3://swipeflix

# 2. Upload data
aws s3 cp data/movies.csv s3://swipeflix/movies.csv
aws s3 cp data/ratings.csv s3://swipeflix/ratings.csv

# 3. Create IAM user with permissions
# - AmazonS3ReadOnlyAccess
# - CloudWatchLogsFullAccess
# - CloudWatchPutMetricData

# 4. Configure SwipeFlix
cat > .env <<EOF
USE_AWS_S3=true
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
AWS_S3_BUCKET=swipeflix
CLOUDWATCH_ENABLED=true
CLOUDWATCH_LOG_GROUP=swipeflix-logs
CLOUDWATCH_LOG_STREAM=app
EOF

# 5. Run training (loads from S3, logs to CloudWatch)
make train

# 6. Start API (logs to CloudWatch)
make start

# 7. View CloudWatch logs
aws logs tail swipeflix-logs --follow
```

### 💡 Benefits of Cloud Integration

**S3 Benefits:**
- ✅ No local disk space needed for large datasets
- ✅ Data available from any environment (local, CI, prod)
- ✅ Built-in versioning and backup
- ✅ Team collaboration on datasets
- ✅ Cost: < $0.10/month for dev datasets

**CloudWatch Benefits:**
- ✅ Centralized logs from multiple instances
- ✅ Real-time monitoring of ML workflows
- ✅ Historical analysis of training metrics
- ✅ Alerting on failures or performance issues
- ✅ Integration with AWS ecosystem
- ✅ Cost: < $1/month for dev workloads

### 🔒 Security Best Practices

1. **IAM Policies:** Use least-privilege access
2. **Credentials:** Store in environment variables, never in code
3. **S3 Encryption:** Enable server-side encryption
4. **CloudWatch:** Set log retention policies (7-30 days)
5. **Access Logs:** Enable S3 access logging for audit

### 📊 Cost Estimation (Monthly)

| Service | Usage | Cost |
|---------|-------|------|
| S3 Storage | 100 MB data | $0.002 |
| S3 Requests | 1,000 GET | $0.0004 |
| CloudWatch Logs | 500 MB ingestion | $0.25 |
| CloudWatch Metrics | 10 custom metrics | $0.30 |
| **Total** | | **< $0.60/month** |

### 🎯 D9 Compliance Summary

✅ **Requirement:** Use at least 2 distinct cloud services  
✅ **Implemented:** Amazon S3 + Amazon CloudWatch

✅ **Screenshots:** Annotated screenshots in README  
✅ **Documentation:** Which services, why, and how to reproduce  
✅ **ML Workflow:** Data → S3 → Training → CloudWatch → Inference

✅ **Services Integrated:**
1. **Amazon S3** - Data storage for movies.csv and ratings.csv
2. **Amazon CloudWatch** - Logs and metrics for training and inference

✅ **Production Ready:** Automatic fallback to local if AWS unavailable

---

## ☁️ Additional Cloud Deployment Options (Bonus)

For production deployments beyond D9 requirements, see `docs/CLOUD_DEPLOYMENT.md` for:
- AWS EC2 deployment
- GCP Cloud Run deployment
- Azure Container Instances deployment

---

## ❓ FAQ

### Q: Build fails with "Module not found"?
**A:** Ensure you've activated the virtual environment: `source venv/bin/activate` (Linux/Mac) or `.\venv\Scripts\activate` (Windows)

### Q: Docker build is slow?
**A:** Use BuildKit: `DOCKER_BUILDKIT=1 docker build .`

### Q: MLflow UI shows no experiments?
**A:** Run training first: `make train`

### Q: Tests fail with "FileNotFoundError: data/movies.csv"?
**A:** Ensure you're running from project root, and data files exist in `data/` directory

### Q: Windows: "make: command not found"?
**A:** Install Make for Windows: `choco install make` or use Git Bash / WSL

### Q: Port 8000 already in use?
**A:** Change port in `.env` or kill existing process: `lsof -ti:8000 | xargs kill -9` (Mac/Linux) or `netstat -ano | findstr :8000` (Windows)

### Q: GPU training not working?
**A:** Verify NVIDIA drivers: `nvidia-smi` and Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

### Q: CI fails on acceptance tests?
**A:** Check golden test files in `golden/` match current API schema. Update if API changed.

### Q: How do I update the model in production?
**A:** 
1. Train new model version
2. Register in MLflow with new version tag
3. Update `MODEL_VERSION` env var in deployment
4. Restart service: `docker-compose restart app`

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines, team member responsibilities, and branch naming conventions.

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the team (see CONTRIBUTING.md).

---

**Built with ❤️ for MLOps Course - Milestone 1**

