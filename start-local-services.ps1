# Start Local Services Script (Windows)
Write-Host "Starting local services..." -ForegroundColor Green

# Check if services are already running
$prometheusRunning = Get-Process -Name prometheus -ErrorAction SilentlyContinue
$grafanaRunning = Get-Process -Name grafana-server -ErrorAction SilentlyContinue

# Start Prometheus
if (-not $prometheusRunning) {
    Write-Host "Starting Prometheus..." -ForegroundColor Yellow
    $prometheusPath = "C:\prometheus\prometheus.exe"
    if (Test-Path $prometheusPath) {
        Start-Process -FilePath $prometheusPath -ArgumentList "--config.file=$PSScriptRoot\monitoring\prometheus\prometheus-local.yml", "--storage.tsdb.path=$PSScriptRoot\data\prometheus"
        Write-Host "Prometheus started" -ForegroundColor Green
    } else {
        Write-Host "Prometheus not found at $prometheusPath" -ForegroundColor Red
        Write-Host "Please install Prometheus or update the path in this script" -ForegroundColor Yellow
        Write-Host "Download from: https://prometheus.io/download/" -ForegroundColor Cyan
    }
} else {
    Write-Host "Prometheus already running" -ForegroundColor Green
}

# Start Grafana
if (-not $grafanaRunning) {
    Write-Host "Starting Grafana..." -ForegroundColor Yellow
    $grafanaPath = "C:\Program Files\GrafanaLabs\grafana\bin\grafana-server.exe"
    if (Test-Path $grafanaPath) {
        Start-Process -FilePath $grafanaPath
        Write-Host "Grafana started" -ForegroundColor Green
    } else {
        Write-Host "Grafana not found at $grafanaPath" -ForegroundColor Red
        Write-Host "Please install Grafana or update the path in this script" -ForegroundColor Yellow
        Write-Host "Download from: https://grafana.com/grafana/download" -ForegroundColor Cyan
    }
} else {
    Write-Host "Grafana already running" -ForegroundColor Green
}

# Start PostgreSQL (usually runs as service)
Write-Host "Checking PostgreSQL..." -ForegroundColor Yellow
$pgService = Get-Service -Name postgresql* -ErrorAction SilentlyContinue
if ($pgService) {
    if ($pgService.Status -ne 'Running') {
        Start-Service $pgService.Name
        Write-Host "PostgreSQL started" -ForegroundColor Green
    } else {
        Write-Host "PostgreSQL already running" -ForegroundColor Green
    }
} else {
    Write-Host "PostgreSQL service not found" -ForegroundColor Yellow
    Write-Host "Install from: https://www.postgresql.org/download/windows/" -ForegroundColor Cyan
}

# Start MinIO
Write-Host "Starting MinIO..." -ForegroundColor Yellow
$minioPath = "C:\minio\minio.exe"
if (Test-Path $minioPath) {
    New-Item -ItemType Directory -Force -Path "$PSScriptRoot\data\minio" | Out-Null
    Start-Process -FilePath $minioPath -ArgumentList "server", "$PSScriptRoot\data\minio", "--console-address", ":9001"
    Write-Host "MinIO started" -ForegroundColor Green
} else {
    Write-Host "MinIO not found at $minioPath" -ForegroundColor Yellow
    Write-Host "You can use AWS S3 or skip MinIO if using local file storage" -ForegroundColor Yellow
    Write-Host "Download from: https://min.io/download" -ForegroundColor Cyan
}

# Start MLflow
Write-Host "Starting MLflow..." -ForegroundColor Yellow
# pragma: allowlist secret
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin"

# Check if MLflow is installed
try {
    $mlflowVersion = & mlflow --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        # pragma: allowlist secret
        Start-Process -NoNewWindow -FilePath "mlflow" -ArgumentList "server", "--backend-store-uri", "postgresql://mlflow:mlflow@localhost:5432/mlflow", "--default-artifact-root", "s3://mlflow/", "--host", "0.0.0.0", "--port", "5000"
        Write-Host "MLflow started" -ForegroundColor Green
    } else {
        Write-Host "MLflow not found. Install with: pip install mlflow" -ForegroundColor Yellow
    }
} catch {
    Write-Host "MLflow not found. Install with: pip install mlflow" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Services Status:" -ForegroundColor Green
Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor Cyan
Write-Host "  - Grafana:    http://localhost:3000 (admin/admin)" -ForegroundColor Cyan
Write-Host "  - MLflow:     http://localhost:5000" -ForegroundColor Cyan
Write-Host "  - MinIO:      http://localhost:9001 (minioadmin/minioadmin)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Evidently runs as a Python service" -ForegroundColor Yellow
Write-Host "Start with: python scripts/start-evidently-local.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your API should be running on: http://localhost:8000" -ForegroundColor Green
