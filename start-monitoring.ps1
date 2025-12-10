# PowerShell script to start monitoring services
Write-Host "Starting monitoring services..." -ForegroundColor Green

# Start monitoring services (without app service)
docker-compose --profile dev up prometheus grafana mlflow evidently postgres minio minio-init -d

Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "Checking services..." -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "Services should be available at:" -ForegroundColor Green
Write-Host "  - Grafana:    http://localhost:3000 (admin/admin)"
Write-Host "  - Prometheus: http://localhost:9090"
Write-Host "  - MLflow:     http://localhost:5000"
Write-Host "  - Evidently:  http://localhost:7000"
Write-Host ""
Write-Host "Your local API server should continue running on http://localhost:8000" -ForegroundColor Cyan
