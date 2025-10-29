# k6 Load Testing for SwipeFlix

Load tests for the SwipeFlix API using [k6](https://k6.io/).

## Prerequisites

Install k6:

### macOS
```bash
brew install k6
```

### Linux
```bash
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

### Windows
```bash
choco install k6
```

Or download from [k6 releases](https://github.com/grafana/k6/releases).

## Running Tests

### Basic Run

```bash
k6 run k6/script.js
```

### Custom API URL

```bash
API_URL=http://localhost:8000 k6 run k6/script.js
```

### Different Load Profiles

#### Smoke Test (2 users)
```bash
k6 run --vus 2 --duration 30s k6/script.js
```

#### Load Test (10 users, 2 minutes)
```bash
k6 run --vus 10 --duration 2m k6/script.js
```

#### Stress Test (50 users, 5 minutes)
```bash
k6 run --vus 50 --duration 5m k6/script.js
```

#### Spike Test
```bash
k6 run --stage 10s:1,1m:100,10s:1 k6/script.js
```

## Test Stages

The default test has the following stages:

1. **Ramp-up**: 30s to reach 10 users
2. **Steady**: 1 min at 10 users
3. **Ramp-up**: 30s to reach 20 users
4. **Steady**: 1 min at 20 users
5. **Ramp-down**: 30s back to 0 users

## SLO Thresholds

The test enforces these Service Level Objectives:

- **p95 latency** < 500ms
- **Error rate** < 1%
- **HTTP failure rate** < 5%

## Test Scenarios

### 1. Health Check (10% of traffic)
- `GET /health`
- Validates service health

### 2. Metadata (5% of traffic)
- `GET /metadata`
- Validates service metadata

### 3. Predictions (85% of traffic)
- `POST /predict`
- Main workload: user recommendations
- Random user IDs and top_k values

## Metrics

### Custom Metrics

- `errors`: Error rate across all requests
- `prediction_latency`: Prediction endpoint latency

### Standard k6 Metrics

- `http_reqs`: Total HTTP requests
- `http_req_duration`: Request duration
- `http_req_failed`: Failed requests
- `iterations`: Total test iterations
- `vus`: Virtual users

## Viewing Results

### Console Output

Results are printed to stdout after the test completes.

### JSON Summary

Results are saved to `k6/results/summary.json`:

```bash
cat k6/results/summary.json | jq .
```

### Cloud Results (k6 Cloud)

```bash
k6 cloud k6/script.js
```

## Integration with CI/CD

### GitHub Actions

The CI pipeline includes load testing for tagged releases:

```yaml
- name: Run k6 load test
  run: |
    API_URL=http://localhost:9000 k6 run k6/script.js
```

## Troubleshooting

### Connection Refused

Ensure the API is running:
```bash
curl http://localhost:8000/health
```

### Timeout Errors

Increase think time or reduce virtual users:
```bash
k6 run --vus 5 k6/script.js
```

### Certificate Errors (HTTPS)

Skip TLS verification (test only):
```bash
k6 run --insecure-skip-tls-verify k6/script.js
```

## Advanced Usage

### Custom Environment Variables

```bash
API_URL=https://api.swipeflix.com \
K6_DURATION=5m \
K6_VUS=50 \
k6 run k6/script.js
```

### Output to InfluxDB

```bash
k6 run --out influxdb=http://localhost:8086/k6 k6/script.js
```

### Output to Prometheus

```bash
k6 run --out experimental-prometheus-rw k6/script.js
```

### Distributed Testing

```bash
# Run on multiple machines
k6 run --execution-segment "0:1/4" k6/script.js  # Machine 1
k6 run --execution-segment "1/4:2/4" k6/script.js  # Machine 2
k6 run --execution-segment "2/4:3/4" k6/script.js  # Machine 3
k6 run --execution-segment "3/4:1" k6/script.js  # Machine 4
```

## Best Practices

1. **Start small**: Begin with smoke tests, gradually increase load
2. **Monitor backend**: Watch CPU, memory, database during tests
3. **Baseline first**: Run tests on stable version before changes
4. **Realistic scenarios**: Use production-like data and patterns
5. **Think time**: Add realistic delays between requests
6. **Ramp-up/down**: Gradually increase/decrease load

## References

- [k6 Documentation](https://k6.io/docs/)
- [k6 Test Types](https://k6.io/docs/test-types/introduction/)
- [k6 Metrics](https://k6.io/docs/using-k6/metrics/)

