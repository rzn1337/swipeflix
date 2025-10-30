# Cloud Deployment Guide

This guide provides examples for deploying SwipeFlix to major cloud providers (AWS, GCP, Azure).

---

## Table of Contents

- [AWS Deployment](#aws-deployment)
- [GCP Deployment](#gcp-deployment)
- [Azure Deployment](#azure-deployment)
- [Multi-Cloud Architecture](#multi-cloud-architecture)

---

## AWS Deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         AWS Cloud                            │
├─────────────────────────────────────────────────────────────┤
│  VPC                                                         │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   ECS Fargate  │  │     S3       │  │   CloudWatch    │ │
│  │  (SwipeFlix)   │  │  (Artifacts) │  │   (Monitoring)  │ │
│  └────────────────┘  └──────────────┘  └─────────────────┘ │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │      RDS       │  │   ECR        │  │   Lambda        │ │
│  │  (PostgreSQL)  │  │  (Images)    │  │  (Serverless)   │ │
│  └────────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Services Used

1. **EC2 / ECS Fargate**: Host FastAPI application
2. **S3**: Store MLflow artifacts and datasets
3. **RDS (PostgreSQL)**: MLflow tracking backend
4. **ECR**: Store Docker images
5. **CloudWatch**: Logging and monitoring
6. **Lambda**: Serverless batch predictions (optional)
7. **CloudFront**: CDN for API (optional)

### Deployment Steps

#### 1. Push Image to ECR

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag swipeflix:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/swipeflix:latest

# Push
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/swipeflix:latest
```

#### 2. Create RDS Instance

```bash
aws rds create-db-instance \
  --db-instance-identifier swipeflix-mlflow-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username mlflow \
  --master-user-password <PASSWORD> \
  --allocated-storage 20
```

#### 3. Create S3 Bucket

```bash
aws s3 mb s3://swipeflix-mlflow-artifacts
```

#### 4. Deploy with ECS Fargate

**Task Definition** (`task-definition.json`):

```json
{
  "family": "swipeflix",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "swipeflix-app",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/swipeflix:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MLFLOW_TRACKING_URI",
          "value": "postgresql://mlflow:PASSWORD@rds-endpoint:5432/mlflow"
        },
        {
          "name": "S3_BUCKET",
          "value": "swipeflix-mlflow-artifacts"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/swipeflix",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Deploy:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json

aws ecs create-service \
  --cluster swipeflix-cluster \
  --service-name swipeflix-service \
  --task-definition swipeflix \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```
---

## Best Practices

1. **Use Infrastructure as Code**: Terraform, CloudFormation, ARM templates
2. **Auto-scaling**: Configure based on CPU/memory/requests
3. **Health Checks**: Implement proper health check endpoints
4. **Secrets Management**: Use cloud secret managers (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
5. **Monitoring**: Set up alerts for errors, latency, costs
6. **Backup**: Regular backups of databases and artifacts
7. **CI/CD**: Automate deployments with GitHub Actions

---

## Next Steps

1. Choose cloud provider based on requirements
2. Provision infrastructure using Terraform
3. Set up CI/CD pipeline for automated deployments
4. Configure monitoring and alerts
5. Perform load testing
6. Document runbooks for common operations