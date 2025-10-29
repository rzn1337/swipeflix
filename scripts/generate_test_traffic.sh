#!/bin/bash
# Generate test traffic for Grafana dashboard demo
# This script makes various API calls to populate metrics

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="${API_URL:-http://localhost:8000}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SwipeFlix Traffic Generator${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "API URL: $API_URL"
echo ""

# Check if API is running
if ! curl -sf "$API_URL/health" > /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: API at $API_URL is not responding${NC}"
    echo "Make sure SwipeFlix is running:"
    echo "  docker-compose --profile dev up -d"
    echo "  # OR"
    echo "  make start"
    exit 1
fi

echo -e "${GREEN}✓ API is running${NC}"
echo ""

# Function to make requests with progress
make_requests() {
    local endpoint=$1
    local count=$2
    local method=$3
    local data=$4
    local delay=$5

    echo -e "${BLUE}Making $count requests to $endpoint...${NC}"

    for i in $(seq 1 $count); do
        if [ "$method" == "POST" ]; then
            curl -s -X POST "$API_URL$endpoint" \
                -H "Content-Type: application/json" \
                -d "$data" > /dev/null
        else
            curl -s "$API_URL$endpoint" > /dev/null
        fi

        # Progress indicator
        if [ $((i % 10)) -eq 0 ]; then
            echo -ne "\rProgress: $i/$count"
        fi

        sleep "$delay"
    done

    echo -e "\r${GREEN}✓ Completed $count requests to $endpoint${NC}"
}

# 1. Health checks (fast, lots of requests)
echo -e "\n${YELLOW}1. Generating health check traffic...${NC}"
make_requests "/health" 100 "GET" "" 0.05

# 2. Metadata requests (medium)
echo -e "\n${YELLOW}2. Generating metadata requests...${NC}"
make_requests "/metadata" 50 "GET" "" 0.1

# 3. Prediction requests (slower, CPU intensive)
echo -e "\n${YELLOW}3. Generating prediction requests...${NC}"
for i in $(seq 1 30); do
    USER_ID=$((RANDOM % 1000 + 1))
    TOP_K=$((RANDOM % 10 + 5))

    curl -s -X POST "$API_URL/predict" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\": \"user_$USER_ID\", \"top_k\": $TOP_K}" > /dev/null

    if [ $((i % 5)) -eq 0 ]; then
        echo -ne "\rProgress: $i/30"
    fi

    sleep 0.3
done
echo -e "\r${GREEN}✓ Completed 30 prediction requests${NC}"

# 4. Mixed traffic pattern (simulate real usage)
echo -e "\n${YELLOW}4. Generating mixed traffic pattern...${NC}"
for i in $(seq 1 50); do
    # Random endpoint
    RAND=$((RANDOM % 10))

    if [ $RAND -lt 6 ]; then
        # 60% health checks
        curl -s "$API_URL/health" > /dev/null
    elif [ $RAND -lt 8 ]; then
        # 20% metadata
        curl -s "$API_URL/metadata" > /dev/null
    else
        # 20% predictions
        USER_ID=$((RANDOM % 1000 + 1))
        curl -s -X POST "$API_URL/predict" \
            -H "Content-Type: application/json" \
            -d "{\"user_id\": \"user_$USER_ID\", \"top_k\": 5}" > /dev/null
    fi

    if [ $((i % 10)) -eq 0 ]; then
        echo -ne "\rProgress: $i/50"
    fi

    sleep 0.1
done
echo -e "\r${GREEN}✓ Completed 50 mixed requests${NC}"

# 5. Burst traffic (test high load)
echo -e "\n${YELLOW}5. Generating burst traffic...${NC}"
echo "Sending rapid requests (burst)..."

for i in $(seq 1 100); do
    curl -s "$API_URL/health" > /dev/null &
done
wait

echo -e "${GREEN}✓ Completed burst traffic (100 concurrent requests)${NC}"

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Traffic generation complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total requests made:"
echo "  - Health checks: ~250"
echo "  - Metadata: 50"
echo "  - Predictions: 30"
echo "  - Mixed: 50"
echo ""
echo "Next steps:"
echo "  1. Open Grafana: http://localhost:3000"
echo "  2. View 'SwipeFlix Monitoring' dashboard"
echo "  3. You should see metrics populated!"
echo ""
echo "To generate continuous traffic:"
echo "  watch -n 5 bash scripts/generate_test_traffic.sh"
echo ""
