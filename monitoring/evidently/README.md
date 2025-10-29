# Evidently Data Drift Monitoring

This directory contains configuration and outputs for [Evidently](https://evidentlyai.com/) data drift monitoring.

## Overview

Evidently monitors:
- **Data Drift**: Distribution changes in features over time
- **Target Drift**: Changes in prediction target distribution
- **Model Performance**: Accuracy, precision, recall degradation
- **Data Quality**: Missing values, type mismatches

## Setup

Evidently is included in Docker Compose:

```bash
docker-compose --profile dev up evidently
```

Access the dashboard at: http://localhost:7000

## Generating Reports

### Python Script

Create a monitoring script (`scripts/generate_evidently_report.py`):

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Load reference and current data
reference_data = pd.read_csv("data/reference.csv")
current_data = pd.read_csv("data/current.csv")

# Generate report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])

report.run(reference_data=reference_data, current_data=current_data)

# Save report
report.save_html("monitoring/evidently/drift_report.html")
```

### Run Monitoring

```bash
python scripts/generate_evidently_report.py
```

## Integration with MLflow

Track drift metrics in MLflow:

```python
import mlflow

# Log drift score
mlflow.log_metric("data_drift_score", drift_score)
mlflow.log_artifact("monitoring/evidently/drift_report.html")
```

## Automated Monitoring

### Scheduled Job

Use cron or Airflow to run drift detection periodically:

```bash
# Run daily at 2 AM
0 2 * * * cd /app && python scripts/generate_evidently_report.py
```

### Real-time Monitoring

For production, integrate with your inference pipeline:

```python
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

# Define tests
tests = TestSuite(tests=[
    TestColumnDrift(column_name='feature_1'),
    TestColumnDrift(column_name='feature_2'),
])

# Run on batch of predictions
tests.run(reference_data=reference, current_data=current)

# Get results
results = tests.as_dict()

if results['failed']:
    # Alert or trigger retraining
    send_alert("Data drift detected!")
```

## Metrics

Key metrics tracked:

- **Drift Score**: 0-1 scale, > 0.1 indicates drift
- **P-value**: Statistical significance of drift
- **Feature Correlation**: Changes in feature relationships
- **Missing Values**: Percentage of missing data

## Alerts

Configure alerts based on thresholds:

```python
if drift_score > 0.15:
    send_slack_alert("High data drift detected!")
    
if missing_values_pct > 5.0:
    send_slack_alert("Data quality issue!")
```

## Best Practices

1. **Baseline Data**: Use recent production data as reference
2. **Regular Updates**: Update reference data periodically
3. **Multiple Windows**: Monitor short-term and long-term drift
4. **Feature Importance**: Focus on important features first
5. **Actionable Alerts**: Set thresholds that trigger retraining

## Resources

- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Drift Detection Guide](https://www.evidentlyai.com/blog/ml-monitoring-data-drift-detection)

