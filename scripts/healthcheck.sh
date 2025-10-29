#!/bin/bash
# Healthcheck script for Docker container

set -e

# Default values
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-/health}"

# Perform health check
response=$(curl -sf "http://${HOST}:${PORT}${ENDPOINT}" || echo "failed")

if [ "$response" = "failed" ]; then
    echo "Health check failed: unable to reach ${HOST}:${PORT}${ENDPOINT}"
    exit 1
fi

# Check if response contains "healthy" status
if echo "$response" | grep -q '"status":"healthy"'; then
    echo "Health check passed"
    exit 0
else
    echo "Health check failed: service not healthy"
    exit 1
fi

