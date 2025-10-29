terraform {
  required_version = ">= 1.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {
  host = var.docker_host
}

# Network for services
resource "docker_network" "swipeflix_network" {
  name = "swipeflix-network"
}

# MinIO Object Storage
resource "docker_image" "minio" {
  name = "minio/minio:latest"
}

resource "docker_container" "minio" {
  name  = "swipeflix-minio-tf"
  image = docker_image.minio.image_id

  command = ["server", "/data", "--console-address", ":9001"]

  ports {
    internal = 9000
    external = var.minio_port
  }

  ports {
    internal = 9001
    external = var.minio_console_port
  }

  env = [
    "MINIO_ROOT_USER=${var.minio_access_key}",
    "MINIO_ROOT_PASSWORD=${var.minio_secret_key}"
  ]

  volumes {
    volume_name    = docker_volume.minio_data.name
    container_path = "/data"
  }

  networks_advanced {
    name = docker_network.swipeflix_network.name
  }

  healthcheck {
    test     = ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
    interval = "10s"
    timeout  = "5s"
    retries  = 5
  }
}

# PostgreSQL for MLflow backend
resource "docker_image" "postgres" {
  name = "postgres:15-alpine"
}

resource "docker_container" "postgres" {
  name  = "swipeflix-postgres-tf"
  image = docker_image.postgres.image_id

  ports {
    internal = 5432
    external = var.postgres_port
  }

  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=${var.postgres_db}"
  ]

  volumes {
    volume_name    = docker_volume.postgres_data.name
    container_path = "/var/lib/postgresql/data"
  }

  networks_advanced {
    name = docker_network.swipeflix_network.name
  }

  healthcheck {
    test     = ["CMD-SHELL", "pg_isready -U ${var.postgres_user}"]
    interval = "10s"
    timeout  = "5s"
    retries  = 5
  }
}

# Volumes
resource "docker_volume" "minio_data" {
  name = "swipeflix-minio-data-tf"
}

resource "docker_volume" "postgres_data" {
  name = "swipeflix-postgres-data-tf"
}

# Outputs
output "minio_endpoint" {
  description = "MinIO API endpoint"
  value       = "http://localhost:${var.minio_port}"
}

output "minio_console_url" {
  description = "MinIO Console URL"
  value       = "http://localhost:${var.minio_console_port}"
}

output "minio_credentials" {
  description = "MinIO credentials"
  value = {
    access_key = var.minio_access_key
    secret_key = var.minio_secret_key
  }
  sensitive = true
}

output "postgres_connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql://${var.postgres_user}:${var.postgres_password}@localhost:${var.postgres_port}/${var.postgres_db}"
  sensitive   = true
}

output "network_name" {
  description = "Docker network name"
  value       = docker_network.swipeflix_network.name
}

