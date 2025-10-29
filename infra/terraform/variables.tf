variable "docker_host" {
  description = "Docker host address"
  type        = string
  default     = "unix:///var/run/docker.sock"
}

variable "minio_port" {
  description = "MinIO API port"
  type        = number
  default     = 9000
}

variable "minio_console_port" {
  description = "MinIO Console port"
  type        = number
  default     = 9001
}

variable "minio_access_key" {
  description = "MinIO access key (username)"
  type        = string
  default     = "minioadmin"
  sensitive   = true
}

variable "minio_secret_key" {
  description = "MinIO secret key (password)"
  type        = string
  default     = "minioadmin"
  sensitive   = true
}

variable "postgres_port" {
  description = "PostgreSQL port"
  type        = number
  default     = 5432
}

variable "postgres_user" {
  description = "PostgreSQL username"
  type        = string
  default     = "mlflow"
}

variable "postgres_password" {
  description = "PostgreSQL password"
  type        = string
  default     = "mlflow"
  sensitive   = true
}

variable "postgres_db" {
  description = "PostgreSQL database name"
  type        = string
  default     = "mlflow"
}
