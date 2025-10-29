"""AWS utilities for S3 and CloudWatch integration."""

import io
import os
from datetime import datetime
from typing import Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from loguru import logger


class AWSManager:
    """Manages AWS S3 and CloudWatch interactions."""

    def __init__(self):
        """Initialize AWS clients."""
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        # Initialize clients only if credentials are available
        self.s3_client = None
        self.cloudwatch_client = None
        self.logs_client = None
        
        if self.aws_access_key and self.aws_secret_key:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                self.logs_client = boto3.client(
                    'logs',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                self.cloudwatch_client = boto3.client(
                    'cloudwatch',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                logger.info("AWS clients initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS clients: {e}")
        else:
            logger.info("AWS credentials not found, running in local mode")

    def is_aws_enabled(self) -> bool:
        """Check if AWS integration is enabled."""
        return self.s3_client is not None

    # ==================== S3 Operations ====================

    def load_csv_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """
        Load CSV file from S3 bucket.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            
        Returns:
            pandas DataFrame with CSV data
            
        Raises:
            Exception if file cannot be loaded
        """
        if not self.s3_client:
            raise Exception("S3 client not initialized. Check AWS credentials.")
        
        try:
            logger.info(f"Loading data from s3://{bucket}/{key}")
            
            # Get object from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            # Read CSV data
            csv_data = response['Body'].read()
            df = pd.read_csv(io.BytesIO(csv_data))
            
            logger.info(f"Successfully loaded {len(df)} rows from S3")
            return df
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404' or error_code == 'NoSuchKey':
                logger.error(f"File not found in S3: s3://{bucket}/{key}")
            else:
                logger.error(f"S3 error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load data from S3: {e}")
            raise

    def upload_file_to_s3(self, file_path: str, bucket: str, key: str) -> bool:
        """
        Upload file to S3 bucket.
        
        Args:
            file_path: Local file path
            bucket: S3 bucket name
            key: S3 object key (destination path)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            logger.warning("S3 client not initialized")
            return False
        
        try:
            logger.info(f"Uploading {file_path} to s3://{bucket}/{key}")
            self.s3_client.upload_file(file_path, bucket, key)
            logger.info("Upload successful")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False

    def check_s3_file_exists(self, bucket: str, key: str) -> bool:
        """
        Check if file exists in S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            True if file exists, False otherwise
        """
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    # ==================== CloudWatch Logs Operations ====================

    def create_log_group_if_not_exists(self, log_group_name: str) -> bool:
        """
        Create CloudWatch log group if it doesn't exist.
        
        Args:
            log_group_name: Name of the log group
            
        Returns:
            True if created or already exists, False on error
        """
        if not self.logs_client:
            logger.debug("CloudWatch Logs client not initialized")
            return False
        
        try:
            self.logs_client.create_log_group(logGroupName=log_group_name)
            logger.info(f"Created CloudWatch log group: {log_group_name}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                logger.debug(f"Log group already exists: {log_group_name}")
                return True
            else:
                logger.error(f"Failed to create log group: {e}")
                return False

    def create_log_stream_if_not_exists(self, log_group_name: str, log_stream_name: str) -> bool:
        """
        Create CloudWatch log stream if it doesn't exist.
        
        Args:
            log_group_name: Name of the log group
            log_stream_name: Name of the log stream
            
        Returns:
            True if created or already exists, False on error
        """
        if not self.logs_client:
            return False
        
        try:
            self.logs_client.create_log_stream(
                logGroupName=log_group_name,
                logStreamName=log_stream_name
            )
            logger.info(f"Created CloudWatch log stream: {log_stream_name}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                logger.debug(f"Log stream already exists: {log_stream_name}")
                return True
            else:
                logger.error(f"Failed to create log stream: {e}")
                return False

    def send_log_to_cloudwatch(
        self,
        log_group_name: str,
        log_stream_name: str,
        message: str,
        timestamp: Optional[int] = None
    ) -> bool:
        """
        Send log message to CloudWatch.
        
        Args:
            log_group_name: CloudWatch log group name
            log_stream_name: CloudWatch log stream name
            message: Log message to send
            timestamp: Unix timestamp in milliseconds (defaults to now)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.logs_client:
            logger.debug(f"CloudWatch not enabled, skipping log: {message}")
            return False
        
        try:
            if timestamp is None:
                timestamp = int(datetime.utcnow().timestamp() * 1000)
            
            # Get sequence token for the stream
            try:
                response = self.logs_client.describe_log_streams(
                    logGroupName=log_group_name,
                    logStreamNamePrefix=log_stream_name,
                    limit=1
                )
                
                sequence_token = None
                if response['logStreams']:
                    sequence_token = response['logStreams'][0].get('uploadSequenceToken')
                
                # Put log events
                kwargs = {
                    'logGroupName': log_group_name,
                    'logStreamName': log_stream_name,
                    'logEvents': [
                        {
                            'timestamp': timestamp,
                            'message': message
                        }
                    ]
                }
                
                if sequence_token:
                    kwargs['sequenceToken'] = sequence_token
                
                self.logs_client.put_log_events(**kwargs)
                logger.debug(f"Sent log to CloudWatch: {message[:50]}...")
                return True
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create log group and stream if they don't exist
                    self.create_log_group_if_not_exists(log_group_name)
                    self.create_log_stream_if_not_exists(log_group_name, log_stream_name)
                    # Retry
                    return self.send_log_to_cloudwatch(log_group_name, log_stream_name, message, timestamp)
                else:
                    raise
                    
        except Exception as e:
            logger.warning(f"Failed to send log to CloudWatch: {e}")
            return False

    def send_metric_to_cloudwatch(
        self,
        namespace: str,
        metric_name: str,
        value: float,
        unit: str = 'None',
        dimensions: Optional[dict] = None
    ) -> bool:
        """
        Send custom metric to CloudWatch.
        
        Args:
            namespace: CloudWatch namespace
            metric_name: Metric name
            value: Metric value
            unit: Metric unit (Seconds, Count, etc.)
            dimensions: Optional dimensions as dict
            
        Returns:
            True if successful, False otherwise
        """
        if not self.cloudwatch_client:
            return False
        
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow()
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            self.cloudwatch_client.put_metric_data(
                Namespace=namespace,
                MetricData=[metric_data]
            )
            
            logger.debug(f"Sent metric to CloudWatch: {metric_name}={value}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to send metric to CloudWatch: {e}")
            return False


# Global AWS manager instance
aws_manager = AWSManager()


# Convenience functions
def load_data_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """
    Load CSV data from S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        pandas DataFrame
    """
    return aws_manager.load_csv_from_s3(bucket, key)


def log_to_cloudwatch(message: str, log_group: str = "swipeflix-logs", log_stream: str = "app"):
    """
    Send log message to CloudWatch.
    
    Args:
        message: Log message
        log_group: CloudWatch log group name
        log_stream: CloudWatch log stream name
    """
    aws_manager.send_log_to_cloudwatch(log_group, log_stream, message)


def send_metric(metric_name: str, value: float, unit: str = 'Count', dimensions: Optional[dict] = None):
    """
    Send metric to CloudWatch.
    
    Args:
        metric_name: Metric name
        value: Metric value
        unit: Metric unit
        dimensions: Optional dimensions
    """
    aws_manager.send_metric_to_cloudwatch(
        namespace="SwipeFlix",
        metric_name=metric_name,
        value=value,
        unit=unit,
        dimensions=dimensions
    )

