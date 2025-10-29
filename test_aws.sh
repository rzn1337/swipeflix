#!/bin/bash
echo "=== Testing AWS Configuration ==="
echo ""
echo "1. Checking if .env exists..."
if [ -f .env ]; then
    echo "✓ .env found"
    echo ""
    echo "2. AWS_S3_DATA_PREFIX value:"
    grep "AWS_S3_DATA_PREFIX" .env || echo "Not set (OK if using root)"
    echo ""
else
    echo "✗ .env NOT FOUND - you need to create it!"
    exit 1
fi

echo "3. Testing AWS S3 access..."
if aws s3 ls s3://swipeflix/ 2>&1 | grep -q "movies.csv"; then
    echo "✓ S3 bucket accessible, movies.csv found"
    aws s3 ls s3://swipeflix/
else
    echo "✗ Cannot access S3 bucket"
    echo "Output:"
    aws s3 ls s3://swipeflix/ 2>&1
fi
