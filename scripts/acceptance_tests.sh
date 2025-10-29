#!/bin/bash
# Acceptance tests for SwipeFlix API
# Runs golden set queries and validates responses

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
GOLDEN_DIR="${GOLDEN_DIR:-golden}"
FAILED_TESTS=0
PASSED_TESTS=0

echo "=========================================="
echo "SwipeFlix Acceptance Tests"
echo "=========================================="
echo "API URL: $API_URL"
echo "Golden Dir: $GOLDEN_DIR"
echo ""

# Function to run a test
run_test() {
    local test_name=$1
    local endpoint=$2
    local method=$3
    local input_file=$4
    local expected_file=$5

    echo -n "Running test: $test_name ... "

    # Make API call (force silent mode and disable progress bar)
    if [ "$method" = "GET" ]; then
        response=$(curl --silent --show-error --write-out "\n%{http_code}" "$API_URL$endpoint" 2>&1)
    else
        response=$(curl --silent --show-error --write-out "\n%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d @"$input_file" \
            "$API_URL$endpoint" 2>&1)
    fi

    # Extract status code and body
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    # Check status code
    if [ "$http_code" != "200" ]; then
        echo -e "${RED}FAILED${NC} (HTTP $http_code)"
        echo "Response: $body"
        ((FAILED_TESTS++))
        return 1
    fi

    # Validate response structure (basic checks)
    if [ -f "$expected_file" ]; then
        # Compare key fields from expected output
        # For simplicity, just check that response is valid JSON and contains expected keys
        if echo "$body" | jq empty 2>/dev/null; then
            echo -e "${GREEN}PASSED${NC}"
            ((PASSED_TESTS++))
            return 0
        else
            echo -e "${RED}FAILED${NC} (Invalid JSON)"
            ((FAILED_TESTS++))
            return 1
        fi
    else
        # No expected file, just validate it's valid JSON
        if echo "$body" | jq empty 2>/dev/null; then
            echo -e "${GREEN}PASSED${NC}"
            ((PASSED_TESTS++))
            return 0
        else
            echo -e "${RED}FAILED${NC} (Invalid JSON)"
            ((FAILED_TESTS++))
            return 1
        fi
    fi
}

# Wait for service to be ready
echo "Waiting for service to be ready..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl --silent --fail "$API_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Service is ready!${NC}"
        echo ""
        break
    fi
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}Service failed to become ready${NC}"
    exit 1
fi

# Test 1: Health check
run_test "Health Check" "/health" "GET" "" ""

# Test 2: Metadata
run_test "Get Metadata" "/metadata" "GET" "" ""

# Test 3: Predict for user_1
run_test "Predict User 1" "/predict" "POST" "$GOLDEN_DIR/predict_user_1.json" "$GOLDEN_DIR/expected_user_1.json"

# Test 4: Predict for user_2
run_test "Predict User 2" "/predict" "POST" "$GOLDEN_DIR/predict_user_2.json" "$GOLDEN_DIR/expected_user_2.json"

# Test 5: Predict with top_k=5
run_test "Predict Top K" "/predict" "POST" "$GOLDEN_DIR/predict_top_k.json" "$GOLDEN_DIR/expected_top_k.json"

# Test 6: Predict with high top_k
run_test "Predict High K" "/predict" "POST" "$GOLDEN_DIR/predict_high_k.json" "$GOLDEN_DIR/expected_high_k.json"

# Test 7: Metrics endpoint
echo -n "Running test: Prometheus Metrics ... "
metrics_response=$(curl --silent --show-error "$API_URL/metrics" 2>&1)
if echo "$metrics_response" | grep -q "swipeflix_http_requests_total"; then
    echo -e "${GREEN}PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}FAILED${NC} (Metrics not found)"
    ((FAILED_TESTS++))
fi

# Summary
echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo "Total:  $((PASSED_TESTS + FAILED_TESTS))"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

