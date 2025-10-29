# Contributing to SwipeFlix

Thank you for your interest in contributing to SwipeFlix! This document provides
guidelines and information for contributors.

______________________________________________________________________

## 👥 Team Members

| Name                | Student ID | Email                         | GitHub Handle |
| ------------------- | ---------- | ----------------------------- | ------------- |
| \[Syed Ali Rizwan\] | 26906      | s.rizwan.26906@khi.iba.edu.pk | @username1    |
| \[Syed Hamza Asif\] | 26975      | s.asif.26975@khi.iba.edu.pk   | @username2    |
| \[Umaima Raheel\]   | ERP-XXXXX  | student3@example.edu          | @username3    |
| \[Sana Arshad\]     | ERP-XXXXX  | student4@example.edu          | @username4    |

______________________________________________________________________

## 📋 Task Distribution

### Member 1: \[Name\] - Data Engineering & ML Pipeline

**Responsibilities:**

- ✅ Data preprocessing and exploratory analysis
- ✅ Implemented hybrid recommender model (collaborative + content-based)
- ✅ MLflow integration for experiment tracking
- ✅ Model training pipeline with deterministic seeding
- ✅ DVC setup for data versioning
- ✅ Feature engineering and TF-IDF vectorization

**Key Files:**

- `src/swipeflix/ml/train.py`
- `src/swipeflix/ml/predict.py`
- `src/swipeflix/ml/preprocessing.py`
- `dvc.yaml`

______________________________________________________________________

### Member 2: \[Name\] - API Development & Service Layer

**Responsibilities:**

- ✅ FastAPI application architecture
- ✅ REST API endpoints (/health, /predict, /metadata, /metrics)
- ✅ Request/response schemas and validation
- ✅ API documentation and examples
- ✅ Uvicorn server configuration
- ✅ Error handling and logging

**Key Files:**

- `src/swipeflix/api/main.py`
- `src/swipeflix/api/routes.py`
- `src/swipeflix/api/schemas.py`
- `src/swipeflix/api/middleware.py`

______________________________________________________________________

### Member 3: \[Name\] - DevOps & CI/CD

**Responsibilities:**

- ✅ Dockerfile (multi-stage build, security hardening)
- ✅ Docker Compose orchestration with profiles
- ✅ GitHub Actions CI/CD pipeline
- ✅ GHCR integration and image publishing
- ✅ Canary deployment and acceptance testing
- ✅ Pre-commit hooks configuration
- ✅ Makefile automation

**Key Files:**

- `Dockerfile`
- `Dockerfile.gpu`
- `docker-compose.yml`
- `.github/workflows/ci.yml`
- `Makefile`
- `.pre-commit-config.yaml`

______________________________________________________________________

### Member 4: \[Name\] - Monitoring & Infrastructure

**Responsibilities:**

- ✅ Prometheus metrics instrumentation
- ✅ Grafana dashboard creation
- ✅ Evidently data drift monitoring
- ✅ Terraform IaC for local MinIO setup
- ✅ k6 load testing scripts
- ✅ Health check and acceptance test scripts
- ✅ Self-hosted GPU runner documentation

**Key Files:**

- `src/swipeflix/monitoring/metrics.py`
- `infra/terraform/`
- `k6/script.js`
- `scripts/acceptance_tests.sh`
- `monitoring/grafana/dashboards/`
- `monitoring/prometheus/prometheus.yml`

______________________________________________________________________

## 🌿 Branch Naming Convention

We follow a structured branch naming convention to keep the repository organized:

### Branch Prefixes

| Prefix      | Purpose                   | Example                            |
| ----------- | ------------------------- | ---------------------------------- |
| `feat/`     | New features              | `feat/add-user-authentication`     |
| `fix/`      | Bug fixes                 | `fix/inference-timeout`            |
| `docs/`     | Documentation updates     | `docs/update-api-examples`         |
| `test/`     | Test additions/updates    | `test/add-integration-tests`       |
| `refactor/` | Code refactoring          | `refactor/simplify-model-loading`  |
| `chore/`    | Maintenance tasks         | `chore/update-dependencies`        |
| `infra/`    | Infrastructure changes    | `infra/add-terraform-modules`      |
| `ci/`       | CI/CD pipeline changes    | `ci/add-gpu-runner-job`            |
| `perf/`     | Performance improvements  | `perf/optimize-inference-pipeline` |
| `hotfix/`   | Critical production fixes | `hotfix/memory-leak`               |

### Branch Naming Examples

```bash
# Good branch names
feat/hybrid-recommender-model
fix/mlflow-connection-error
docs/add-cloud-deployment-guide
infra/setup-minio-terraform
ci/add-load-testing-job

# Bad branch names
new-feature
fix
update
john-working-branch
```

### Workflow

1. **Create branch from main:**

   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/your-feature-name
   ```

1. **Make commits with descriptive messages:**

   ```bash
   git add .
   git commit -m "feat: add user-based collaborative filtering"
   ```

1. **Push and create Pull Request:**

   ```bash
   git push origin feat/your-feature-name
   ```

1. **PR must pass all CI checks before merge**

______________________________________________________________________

## 📝 Commit Message Convention

We follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `perf`: Performance improvements

### Examples

```bash
feat(api): add /predict endpoint with batch support

Implemented batch prediction endpoint that accepts multiple user IDs
and returns recommendations for all users in a single request.

Closes #123

---

fix(ml): resolve memory leak in model training

Training pipeline was not releasing memory after each epoch.
Added explicit garbage collection and tensor cleanup.

Fixes #456

---

docs(readme): update quick start instructions

Added Windows-specific setup steps and common troubleshooting
tips for Docker Compose.
```

______________________________________________________________________

## 🔄 Pull Request Process

1. **Ensure your PR:**

   - Has a descriptive title and description
   - References related issues (e.g., "Closes #123")
   - Passes all CI checks (lint, test, build)
   - Has test coverage ≥80%
   - Updates documentation if needed

1. **PR Template:**

   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Comments added for complex logic
   - [ ] Documentation updated
   - [ ] No new warnings generated
   - [ ] Tests pass locally
   ```

1. **Review Process:**

   - At least 1 approval required
   - All CI checks must pass
   - No merge conflicts
   - Squash and merge preferred for feature branches

______________________________________________________________________

## 🧪 Testing Guidelines

### Running Tests Locally

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage report
pytest --cov=src/swipeflix --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Writing Tests

- **Unit tests:** Test individual functions/classes in isolation
- **Integration tests:** Test component interactions
- **Acceptance tests:** End-to-end golden set queries

Example:

```python
def test_predict_endpoint_returns_recommendations():
    """Test that /predict returns valid recommendations."""
    response = client.post("/predict", json={"user_id": "user_1", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5
```

______________________________________________________________________

## 🎨 Code Style

### Python Style Guide

- **Formatter:** Black (line length 88)
- **Linter:** Ruff
- **Type hints:** Encouraged for public APIs
- **Docstrings:** Google style

```python
def predict_for_user(user_id: str, top_k: int = 10) -> list[dict]:
    """Get top-K movie recommendations for a user.

    Args:
        user_id: User identifier
        top_k: Number of recommendations to return

    Returns:
        List of recommendation dictionaries with movie_id, title, and score

    Raises:
        ValueError: If user_id is not found
    """
    pass
```

### Pre-commit Hooks

Run before every commit:

```bash
make precommit
```

______________________________________________________________________

## 🐛 Reporting Issues

### Bug Reports

Use the bug report template and include:

- **Description:** Clear description of the bug
- **Steps to Reproduce:** Minimal steps to reproduce
- **Expected Behavior:** What should happen
- **Actual Behavior:** What actually happens
- **Environment:** OS, Python version, Docker version
- **Logs:** Relevant error messages

### Feature Requests

Use the feature request template and include:

- **Problem:** What problem does this solve?
- **Proposed Solution:** How would you implement it?
- **Alternatives:** Other solutions considered
- **Additional Context:** Screenshots, examples, etc.

______________________________________________________________________

## 📚 Documentation Standards

- **README:** High-level overview, quick start, architecture
- **Code comments:** Explain "why", not "what"
- **Docstrings:** All public functions and classes
- **API docs:** Automatically generated from FastAPI
- **Architecture decisions:** Document significant design choices

______________________________________________________________________

## 🔒 Security

- **Never commit secrets:** Use environment variables
- **Dependency scanning:** CI runs pip-audit
- **Code review:** Security-focused review for auth/sensitive code
- **Responsible disclosure:** Report security issues privately

______________________________________________________________________

## 📜 Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for our community guidelines.

______________________________________________________________________

## 📞 Getting Help

- **GitHub Issues:** For bug reports and feature requests
- **GitHub Discussions:** For questions and general discussion
- **Team Chat:** \[Your team communication channel\]

______________________________________________________________________

## 🎓 Learning Resources

### MLOps

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI](https://docs.evidentlyai.com/)

### FastAPI

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Models](https://docs.pydantic.dev/)

### Docker & Kubernetes

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Compose](https://docs.docker.com/compose/)

### CI/CD

- [GitHub Actions](https://docs.github.com/en/actions)
- [k6 Load Testing](https://k6.io/docs/)

______________________________________________________________________

Thank you for contributing to SwipeFlix! 🎬🍿
