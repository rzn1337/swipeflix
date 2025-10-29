# SwipeFlix Terraform Infrastructure

This Terraform configuration provisions local infrastructure for SwipeFlix using Docker
containers.

## Components

- **MinIO**: S3-compatible object storage for MLflow artifacts
- **PostgreSQL**: Database backend for MLflow tracking server
- **Docker Network**: Isolated network for service communication

## Prerequisites

1. **Terraform** >= 1.0

   ```bash
   # macOS
   brew install terraform

   # Linux
   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
   unzip terraform_1.6.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   ```

1. **Docker** running locally

## Quick Start

### 1. Initialize Terraform

```bash
cd infra/terraform
terraform init
```

### 2. Review the Plan

```bash
terraform plan
```

### 3. Apply Configuration

```bash
# Using default values
terraform apply

# Or with custom values
terraform apply \
  -var="minio_access_key=myaccesskey" \
  -var="minio_secret_key=mysecretkey"
```

### 4. View Outputs

```bash
terraform output
```

Example output:

```
minio_console_url = "http://localhost:9001"
minio_endpoint = "http://localhost:9000"
network_name = "swipeflix-network"
postgres_connection_string = <sensitive>
```

To view sensitive outputs:

```bash
terraform output -json | jq .
```

## Configuration

### Using terraform.tfvars

Create a `terraform.tfvars` file (gitignored):

```hcl
minio_access_key    = "minioadmin"
minio_secret_key    = "minioadmin"
postgres_user       = "mlflow"
postgres_password   = "mlflow"
postgres_db         = "mlflow"
```

Then apply:

```bash
terraform apply
```

### Using Environment Variables

```bash
export TF_VAR_minio_access_key="minioadmin"
export TF_VAR_minio_secret_key="minioadmin"
terraform apply
```

## Accessing Services

### MinIO Console

- URL: http://localhost:9001
- Username: `minioadmin` (or your custom value)
- Password: `minioadmin` (or your custom value)

### MinIO API

- Endpoint: http://localhost:9000
- Use with AWS CLI or boto3

### PostgreSQL

- Host: localhost
- Port: 5432
- Database: mlflow
- Username: mlflow
- Password: mlflow

## Integration with Docker Compose

The Terraform-managed containers use the same network name as Docker Compose, allowing
seamless integration:

```bash
# Start Terraform infrastructure
cd infra/terraform
terraform apply -auto-approve

# Start Docker Compose services (will connect to Terraform network)
cd ../..
docker-compose --profile dev up
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

Or for non-interactive mode:

```bash
terraform destroy -auto-approve
```

## Troubleshooting

### Port Already in Use

If you get port conflict errors:

```bash
# Check what's using the port
lsof -ti:9000  # MinIO API
lsof -ti:9001  # MinIO Console
lsof -ti:5432  # PostgreSQL

# Kill the process or change ports in variables
terraform apply -var="minio_port=9002"
```

### Docker Connection Issues

```bash
# Verify Docker is running
docker ps

# Check Docker socket permissions
ls -l /var/run/docker.sock
```

### State Lock Issues

If Terraform state is locked:

```bash
# Force unlock (use with caution)
terraform force-unlock <LOCK_ID>
```

## Advanced Usage

### Remote State Backend

For team collaboration, use a remote backend (e.g., S3):

```hcl
terraform {
  backend "s3" {
    bucket = "swipeflix-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
  }
}
```

### Modules

This configuration can be converted into a reusable module:

```hcl
module "swipeflix_infra" {
  source = "./infra/terraform"

  minio_access_key = "custom_key"
  postgres_password = "secure_password"
}
```

## Security Notes

- **Never commit** `terraform.tfvars` or `.tfstate` files
- Use strong passwords in production
- Consider using HashiCorp Vault for secret management
- Restrict network access in production environments

## Testing

To test the infrastructure:

```bash
# Apply configuration
terraform apply -auto-approve

# Verify MinIO is accessible
curl http://localhost:9000/minio/health/live

# Verify PostgreSQL is accessible
pg_isready -h localhost -p 5432 -U mlflow

# Cleanup
terraform destroy -auto-approve
```

## CI/CD Integration

GitHub Actions example:

```yaml
- name: Setup Terraform
  uses: hashicorp/setup-terraform@v2

- name: Terraform Init
  run: terraform init
  working-directory: infra/terraform

- name: Terraform Apply
  run: terraform apply -auto-approve
  working-directory: infra/terraform
  env:
    TF_VAR_minio_access_key: ${{ secrets.MINIO_ACCESS_KEY }}
    TF_VAR_minio_secret_key: ${{ secrets.MINIO_SECRET_KEY }}
```

## Support

For issues or questions, see the main [README](../../README.md) or open a GitHub issue.
