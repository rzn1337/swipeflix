#!/bin/bash
# Setup AWS Integration for SwipeFlix (D9 - Cloud Integration)

set -e

echo "=========================================="
echo "SwipeFlix AWS Integration Setup"
echo "=========================================="
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first:"
    echo "   https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    exit 1
fi

echo "✅ AWS CLI found"
echo ""

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run: aws configure"
    exit 1
fi

echo "✅ AWS credentials configured"
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "   Account: $AWS_ACCOUNT"
echo ""

# Configuration
BUCKET_NAME="${AWS_S3_BUCKET:-swipeflix}"
AWS_REGION="${AWS_REGION:-us-east-1}"
LOG_GROUP="${CLOUDWATCH_LOG_GROUP:-swipeflix-logs}"

echo "Configuration:"
echo "  S3 Bucket: $BUCKET_NAME"
echo "  Region: $AWS_REGION"
echo "  Log Group: $LOG_GROUP"
echo ""

# Create S3 bucket
echo "Step 1: Creating S3 bucket..."
if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
    aws s3 mb "s3://$BUCKET_NAME" --region "$AWS_REGION"
    echo "✅ Created S3 bucket: $BUCKET_NAME"
else
    echo "✅ S3 bucket already exists: $BUCKET_NAME"
fi
echo ""

# Upload data files
echo "Step 2: Uploading data files to S3..."
if [ -f "data/movies.csv" ]; then
    aws s3 cp data/movies.csv "s3://$BUCKET_NAME/movies.csv"
    echo "✅ Uploaded movies.csv"
else
    echo "⚠️  data/movies.csv not found, skipping"
fi

if [ -f "data/ratings.csv" ]; then
    aws s3 cp data/ratings.csv "s3://$BUCKET_NAME/ratings.csv"
    echo "✅ Uploaded ratings.csv"
else
    echo "⚠️  data/ratings.csv not found, skipping"
fi
echo ""

# Verify uploads
echo "Step 3: Verifying S3 uploads..."
aws s3 ls "s3://$BUCKET_NAME/"
echo ""

# Create CloudWatch log group
echo "Step 4: Creating CloudWatch log group..."
if aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --region "$AWS_REGION" 2>&1 | grep -q "$LOG_GROUP"; then
    echo "✅ CloudWatch log group already exists: $LOG_GROUP"
else
    aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$AWS_REGION"
    echo "✅ Created CloudWatch log group: $LOG_GROUP"
fi
echo ""

# Create .env file
echo "Step 5: Creating .env configuration..."
cat > .env.aws <<EOF
# AWS Integration Configuration (D9)
USE_AWS_S3=true
AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
AWS_REGION=$AWS_REGION
AWS_S3_BUCKET=$BUCKET_NAME

# CloudWatch Logging
CLOUDWATCH_ENABLED=true
CLOUDWATCH_LOG_GROUP=$LOG_GROUP
CLOUDWATCH_LOG_STREAM=app

# Other settings (copy from your existing .env)
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=SwipeFlixModel
MODEL_VERSION=1
EOF

echo "✅ Created .env.aws configuration file"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review .env.aws and merge with your .env file"
echo "2. Run training: make train"
echo "3. View CloudWatch logs: aws logs tail $LOG_GROUP --follow"
echo "4. View S3 bucket: aws s3 ls s3://$BUCKET_NAME/"
echo ""
echo "📸 For D9 screenshots, visit AWS Console:"
echo "  - S3: https://console.aws.amazon.com/s3/buckets/$BUCKET_NAME"
echo "  - CloudWatch: https://console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#logsV2:log-groups/log-group/$LOG_GROUP"
echo ""
echo "✅ AWS Integration Ready!"
