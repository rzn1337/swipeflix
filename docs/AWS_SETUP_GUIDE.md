# AWS Integration Setup Guide (D9 - Cloud Integration)

This guide explains how to set up and use AWS S3 and CloudWatch for SwipeFlix.

## Overview

SwipeFlix integrates with two AWS services to meet D9 Cloud Integration requirements:

1. **Amazon S3** - Data storage for movies.csv and ratings.csv
2. **Amazon CloudWatch** - Centralized logging and metrics monitoring

---

## Prerequisites

- AWS Account
- AWS CLI installed (optional but recommended)
- AWS Access Key ID and Secret Access Key with appropriate permissions

---

## Part 1: AWS S3 Setup

### Step 1: Create S3 Bucket

```bash
# Using AWS CLI
aws s3 mb s3://swipeflix --region us-east-1

# Or use AWS Console:
# 1. Go to S3 Console
# 2. Click "Create bucket"
# 3. Bucket name: swipeflix
# 4. Region: us-east-1
# 5. Keep default settings
# 6. Click "Create bucket"
```

### Step 2: Upload Data Files

```bash
# Upload from local data directory
aws s3 cp data/movies.csv s3://swipeflix/movies.csv
aws s3 cp data/ratings.csv s3://swipeflix/ratings.csv

# Verify upload
aws s3 ls s3://swipeflix/
# Should show:
# movies.csv
# ratings.csv
```

### Step 3: Set Bucket Permissions (Optional)

For public read access (if needed):

```bash
# Make bucket publicly readable
aws s3api put-bucket-acl --bucket swipeflix --acl public-read

# Or use bucket policy for specific files
aws s3api put-object-acl --bucket swipeflix --key movies.csv --acl public-read
aws s3api put-object-acl --bucket swipeflix --key ratings.csv --acl public-read
```

### Step 4: Create IAM User with S3 Access

1. Go to IAM Console → Users → Create user
2. User name: `swipeflix-app`
3. Attach policies:
   - `AmazonS3ReadOnlyAccess` (for reading data)
   - Or custom policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::swipeflix",
        "arn:aws:s3:::swipeflix/*"
      ]
    }
  ]
}
```

4. Create access key → Save Access Key ID and Secret Access Key

---

## Part 2: AWS CloudWatch Setup

### Step 1: Grant CloudWatch Permissions

Add CloudWatch permissions to the IAM user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*"
    }
  ]
}
```

### Step 2: CloudWatch Log Groups (Auto-Created)

SwipeFlix automatically creates:
- **Log Group**: `swipeflix-logs`
- **Log Stream**: `app`

No manual setup required! The application will create these on first run.

---

## Part 3: Configure SwipeFlix

### Step 1: Set Environment Variables

Create or update `.env` file:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# Enable S3 Integration
USE_AWS_S3=true
AWS_S3_BUCKET=swipeflix
AWS_S3_DATA_PREFIX=  # Leave empty if files are in root

# Enable CloudWatch
CLOUDWATCH_ENABLED=true
CLOUDWATCH_LOG_GROUP=swipeflix-logs
CLOUDWATCH_LOG_STREAM=app
```

### Step 2: Update docker-compose.yml (if using Docker)

Add environment variables to the `app` service:

```yaml
app:
  environment:
    - USE_AWS_S3=true
    - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
    - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    - AWS_REGION=${AWS_REGION}
    - CLOUDWATCH_ENABLED=true
```

---

## Part 4: Verify Integration

### Test S3 Data Loading

```bash
# Run training (will load from S3)
make train

# Expected log output:
# Loading movies data from AWS S3: s3://swipeflix/movies.csv
# Successfully loaded XXXX movies from AWS S3
# Loading ratings data from AWS S3: s3://swipeflix/ratings.csv
# Successfully loaded XXXX ratings from AWS S3
```

### Test CloudWatch Logging

```bash
# Start the API
make start

# Or with Docker
docker-compose up app

# Check CloudWatch logs
aws logs tail swipeflix-logs --follow

# Expected output:
# SwipeFlix API started successfully. Version: 1.0.0, Environment: development, S3 Integration: Enabled
```

### View CloudWatch in AWS Console

1. Go to CloudWatch Console → Logs → Log groups
2. Find `swipeflix-logs`
3. Click on log stream `app`
4. You should see:
   - API startup messages
   - Training events
   - Prediction requests
   - Shutdown messages

### View CloudWatch Metrics

1. Go to CloudWatch Console → Metrics → All metrics
2. Find namespace: `SwipeFlix`
3. Available metrics:
   - `TrainingCompleted` - Number of training runs
   - `TrainingMSE` - Model MSE values
   - `PredictionRequests` - API prediction count
   - `InferenceLatency` - Prediction latency

---

## ML Workflow with AWS

### Data Flow

```
1. Training:
   AWS S3 (movies.csv, ratings.csv)
   ↓
   SwipeFlix Training Script (data_loader.py)
   ↓
   Model Training
   ↓
   MLflow Registry (local or S3)
   ↓
   CloudWatch Logs (training completion)

2. Inference:
   User Request
   ↓
   FastAPI /predict endpoint
   ↓
   Load Model from MLflow
   ↓
   Generate Predictions
   ↓
   CloudWatch Logs (prediction event)
   ↓
   CloudWatch Metrics (latency, count)
   ↓
   Response to User
```

### Benefits of AWS Integration

1. **S3 for Data Storage:**
   - Centralized data management
   - Versioning support
   - High availability
   - Easy sharing across teams
   - Cost-effective storage

2. **CloudWatch for Monitoring:**
   - Centralized logging
   - Real-time monitoring
   - Custom metrics
   - Alerting capabilities
   - Historical analysis

---

## Troubleshooting

### Issue: S3 Access Denied

**Error:** `botocore.exceptions.ClientError: An error occurred (AccessDenied)`

**Solution:**
- Verify IAM user has S3 read permissions
- Check bucket name is correct
- Verify AWS credentials in `.env`

### Issue: CloudWatch Permission Error

**Error:** `AccessDeniedException when calling PutLogEvents`

**Solution:**
- Add CloudWatch permissions to IAM user
- Verify log group name matches configuration
- Check AWS region is correct

### Issue: Files Not Found in S3

**Error:** `File not found in S3: s3://swipeflix/movies.csv`

**Solution:**
```bash
# Verify files exist
aws s3 ls s3://swipeflix/

# Re-upload if needed
aws s3 cp data/movies.csv s3://swipeflix/movies.csv
aws s3 cp data/ratings.csv s3://swipeflix/ratings.csv
```

### Issue: CloudWatch Logs Not Appearing

**Check:**
1. `CLOUDWATCH_ENABLED=true` in `.env`
2. AWS credentials are valid
3. IAM user has CloudWatch permissions
4. Log group/stream names match configuration


---

## Production Recommendations

1. **Use S3 Lifecycle Policies** to archive old data
2. **Enable S3 Versioning** for data recovery
3. **Set up CloudWatch Alarms** for critical events
4. **Use IAM Roles** instead of access keys (in production)
5. **Enable CloudWatch Log Insights** for log analysis
6. **Set log retention policies** to manage costs
7. **Use S3 server-side encryption** for sensitive data

---

## Cleanup

To remove AWS resources:

```bash
# Delete S3 files
aws s3 rm s3://swipeflix/movies.csv
aws s3 rm s3://swipeflix/ratings.csv

# Delete S3 bucket
aws s3 rb s3://swipeflix --force

# Delete CloudWatch log group
aws logs delete-log-group --log-group-name swipeflix-logs

# Delete IAM user
# (Do this manually in AWS Console)
```

---

## References

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [AWS CloudWatch Documentation](https://docs.aws.amazon.com/cloudwatch/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)