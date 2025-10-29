# Screenshots Directory

This directory contains screenshots demonstrating the ML monitoring stack.

## Required Screenshots

### 1. MLflow Model Registry

**Filename:** `mlflow-registry.png`

**What to capture:**

- Navigate to: http://localhost:5000
- Click on "Models" tab
- Show `SwipeFlixModel` with version 1 registered
- Should display:
  - Model name and version
  - Training parameters (n_components, weights, etc.)
  - Metrics (train_mse)
  - Timestamp

**How to generate:**

```bash
make compose-up
make train
# Wait 10 seconds, then visit http://localhost:5000
```

______________________________________________________________________

### 2. Grafana Dashboard

**Filename:** `grafana-dashboard.png`

**What to capture:**

- Navigate to: http://localhost:3000 (admin/admin)
- Go to Dashboards → SwipeFlix API Dashboard
- Should show:
  - Request rate graph
  - Latency metrics (p95, p99)
  - Error rate gauge
  - Model inference latency
  - GPU utilization (if GPU available)

**How to generate:**

```bash
make compose-up
# Generate some traffic
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "top_k": 5}'

# Visit http://localhost:3000
# Login: admin/admin
# Navigate to SwipeFlix API Dashboard
```

______________________________________________________________________

### 3. Evidently Drift Report

**Filename:** `evidently-drift.png`

**What to capture:**

- Run drift detection script
- Open generated HTML report
- Should show:
  - Drift score
  - Feature distribution comparisons
  - Data quality metrics
  - Warnings/alerts if drift detected

**How to generate:**

```bash
python scripts/generate_drift_report.py
open monitoring/evidently/drift_report.html
```

______________________________________________________________________

### 4. Prometheus Metrics (Optional)

**Filename:** `prometheus-metrics.png`

**What to capture:**

- Navigate to: http://localhost:9090
- Show query: `swipeflix_http_requests_total`
- Display metrics in table or graph format

______________________________________________________________________

## Quick Screenshot Generation Script

```bash
#!/bin/bash
# scripts/generate_all_screenshots.sh

echo "Starting services..."
make compose-up

echo "Waiting for services to be ready..."
sleep 30

echo "Training model..."
make train

echo "Generating traffic..."
for i in {1..50}; do
    curl -s http://localhost:8000/health > /dev/null
    curl -s -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d "{\"user_id\": \"user_$i\", \"top_k\": 5}" > /dev/null
done

echo "Generating drift report..."
python scripts/generate_drift_report.py

echo "✅ Ready for screenshots!"
echo ""
echo "1. MLflow: http://localhost:5000"
echo "2. Grafana: http://localhost:3000 (admin/admin)"
echo "3. Evidently: open monitoring/evidently/drift_report.html"
echo "4. Prometheus: http://localhost:9090"
```

______________________________________________________________________

## Automated Screenshot Capture (Optional)

For automated screenshot capture, use tools like:

### Puppeteer (Node.js)

```javascript
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // MLflow
  await page.goto('http://localhost:5000');
  await page.screenshot({ path: 'docs/screenshots/mlflow-registry.png' });

  // Grafana
  await page.goto('http://localhost:3000');
  await page.type('#username', 'admin');
  await page.type('#password', 'admin');
  await page.click('button[type="submit"]');
  await page.waitForNavigation();
  await page.screenshot({ path: 'docs/screenshots/grafana-dashboard.png' });

  await browser.close();
})();
```

### Playwright (Python)

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    # MLflow
    page.goto("http://localhost:5000")
    page.screenshot(path="docs/screenshots/mlflow-registry.png")

    browser.close()
```

______________________________________________________________________

## Notes

- Screenshots should be high resolution (at least 1920x1080)
- Capture full browser window or relevant section
- Ensure all labels and metrics are visible
- Remove any sensitive information before committing
- Use PNG format for better quality

______________________________________________________________________

**Status:**

- [ ] mlflow-registry.png
- [ ] grafana-dashboard.png
- [ ] evidently-drift.png
- [ ] prometheus-metrics.png (optional)
