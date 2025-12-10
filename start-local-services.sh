#!/bin/bash
# Start Local Services Script (Linux/Mac)
echo "Starting local services..."

# Start Prometheus
if ! pgrep -x "prometheus" > /dev/null; then
    echo "Starting Prometheus..."
    prometheus --config.file=monitoring/prometheus/prometheus-local.yml \
               --storage.tsdb.path=data/prometheus &
    echo "Prometheus started (PID: $!)"
else
    echo "Prometheus already running"
fi

# Start Grafana
if ! pgrep -x "grafana-server" > /dev/null; then
    echo "Starting Grafana..."
    grafana-server &
    echo "Grafana started (PID: $!)"
else
    echo "Grafana already running"
fi

# Start PostgreSQL (usually runs as service)
if command -v systemctl &> /dev/null; then
    if systemctl is-active --quiet postgresql; then
        echo "PostgreSQL already running"
    else
        sudo systemctl start postgresql
        echo "PostgreSQL service started"
    fi
elif command -v brew &> /dev/null; then
    # macOS with Homebrew
    brew services start postgresql@15 2>/dev/null || brew services start postgresql 2>/dev/null
    echo "PostgreSQL service started"
fi

# Start MinIO
if ! pgrep -x "minio" > /dev/null; then
    echo "Starting MinIO..."
    mkdir -p data/minio
    minio server data/minio --console-address ":9001" &
    echo "MinIO started (PID: $!)"
else
    echo "MinIO already running"
fi

# Start MLflow
if ! pgrep -f "mlflow server" > /dev/null; then
    echo "Starting MLflow..."
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin

    if command -v mlflow &> /dev/null; then
        # pragma: allowlist secret
        mlflow server \
            --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow \
            --default-artifact-root s3://mlflow/ \
            --host 0.0.0.0 \
            --port 5000 &
        echo "MLflow started (PID: $!)"
    else
        echo "MLflow not found. Install with: pip install mlflow"
    fi
else
    echo "MLflow already running"
fi

echo ""
echo "Services Status:"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana:    http://localhost:3000 (admin/admin)"
echo "  - MLflow:     http://localhost:5000"
echo "  - MinIO:      http://localhost:9001 (minioadmin/minioadmin)"
echo ""
echo "Note: Evidently runs as a Python service"
echo "Start with: python scripts/start-evidently-local.py"
echo ""
echo "Your API should be running on: http://localhost:8000"
