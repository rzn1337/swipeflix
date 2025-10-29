import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const predictionLatency = new Trend('prediction_latency');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp up to 10 users
    { duration: '1m', target: 10 },   // Stay at 10 users
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down to 0
  ],
  thresholds: {
    // SLO: p95 latency < 500ms
    'http_req_duration': ['p(95)<500'],
    // SLO: Error rate < 1%
    'errors': ['rate<0.01'],
    // SLO: 95% of requests should succeed
    'http_req_failed': ['rate<0.05'],
  },
};

// Base URL
const BASE_URL = __ENV.API_URL || 'http://localhost:8000';

// Sample user IDs for testing
const USER_IDS = [
  'user_1', 'user_2', 'user_3', 'user_4', 'user_5',
  'user_10', 'user_20', 'user_30', 'user_40', 'user_50'
];

// Get random user ID
function getRandomUserId() {
  return USER_IDS[Math.floor(Math.random() * USER_IDS.length)];
}

// Setup: Run once before all tests
export function setup() {
  console.log(`Load testing SwipeFlix API at ${BASE_URL}`);
  
  // Check if service is healthy
  const healthRes = http.get(`${BASE_URL}/health`);
  check(healthRes, {
    'setup: service is healthy': (r) => r.status === 200,
  });
  
  return { startTime: new Date().toISOString() };
}

// Main test function
export default function (data) {
  // Test 1: Health Check (10% of requests)
  if (Math.random() < 0.1) {
    const healthRes = http.get(`${BASE_URL}/health`);
    
    check(healthRes, {
      'health check: status 200': (r) => r.status === 200,
      'health check: is healthy': (r) => JSON.parse(r.body).status === 'healthy',
    });
    
    errorRate.add(healthRes.status !== 200);
  }
  
  // Test 2: Metadata (5% of requests)
  if (Math.random() < 0.05) {
    const metadataRes = http.get(`${BASE_URL}/metadata`);
    
    check(metadataRes, {
      'metadata: status 200': (r) => r.status === 200,
      'metadata: has app_name': (r) => JSON.parse(r.body).app_name === 'SwipeFlix',
    });
    
    errorRate.add(metadataRes.status !== 200);
  }
  
  // Test 3: Prediction (85% of requests - main workload)
  const userId = getRandomUserId();
  const topK = Math.floor(Math.random() * 10) + 1; // Random between 1-10
  
  const payload = JSON.stringify({
    user_id: userId,
    top_k: topK,
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const startTime = Date.now();
  const predictionRes = http.post(`${BASE_URL}/predict`, payload, params);
  const duration = Date.now() - startTime;
  
  predictionLatency.add(duration);
  
  const checkResult = check(predictionRes, {
    'prediction: status 200': (r) => r.status === 200,
    'prediction: has recommendations': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body.recommendations);
      } catch {
        return false;
      }
    },
    'prediction: correct user_id': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.user_id === userId;
      } catch {
        return false;
      }
    },
    'prediction: latency < 1s': (r) => r.timings.duration < 1000,
  });
  
  errorRate.add(!checkResult || predictionRes.status !== 200);
  
  // Think time between requests
  sleep(Math.random() * 2 + 1); // Random 1-3 seconds
}

// Teardown: Run once after all tests
export function teardown(data) {
  console.log(`Load test completed. Started at: ${data.startTime}`);
  
  // Final health check
  const healthRes = http.get(`${BASE_URL}/health`);
  check(healthRes, {
    'teardown: service still healthy': (r) => r.status === 200,
  });
}

// Handle summary to export results
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'k6/results/summary.json': JSON.stringify(data),
  };
}

// Helper function for text summary
function textSummary(data, options) {
  const indent = options.indent || '';
  const enableColors = options.enableColors || false;
  
  let summary = '\n';
  summary += `${indent}========== Load Test Summary ==========\n`;
  summary += `${indent}Total Requests: ${data.metrics.http_reqs.values.count}\n`;
  summary += `${indent}Failed Requests: ${data.metrics.http_req_failed.values.passes}\n`;
  summary += `${indent}Request Duration (p95): ${data.metrics.http_req_duration.values['p(95)']}ms\n`;
  summary += `${indent}Request Duration (p99): ${data.metrics.http_req_duration.values['p(99)']}ms\n`;
  summary += `${indent}Error Rate: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%\n`;
  summary += `${indent}=======================================\n`;
  
  return summary;
}

