"""GPU metrics collection for Prometheus."""

import subprocess
from typing import Optional

from loguru import logger
from prometheus_client import Gauge

# GPU Metrics
gpu_utilization = Gauge(
    "swipeflix_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
)

gpu_memory_used = Gauge(
    "swipeflix_gpu_memory_used_mb",
    "GPU memory used in MB",
    ["gpu_id"],
)

gpu_memory_total = Gauge(
    "swipeflix_gpu_memory_total_mb",
    "GPU memory total in MB",
    ["gpu_id"],
)

gpu_temperature = Gauge(
    "swipeflix_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_id"],
)

gpu_power_usage = Gauge(
    "swipeflix_gpu_power_usage_watts",
    "GPU power usage in Watts",
    ["gpu_id"],
)


def check_gpu_available() -> bool:
    """Check if nvidia-smi is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def update_gpu_metrics() -> None:
    """Update GPU metrics from nvidia-smi."""
    if not check_gpu_available():
        logger.debug("GPU not available, skipping metrics update")
        return
    
    try:
        # Query nvidia-smi for metrics
        # Format: gpu_id, utilization.gpu, memory.used, memory.total, temperature.gpu, power.draw
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode != 0:
            logger.warning(f"nvidia-smi failed: {result.stderr}")
            return
        
        # Parse output
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            
            gpu_id = parts[0]
            util = float(parts[1])
            mem_used = float(parts[2])
            mem_total = float(parts[3])
            temp = float(parts[4])
            power = float(parts[5])
            
            # Update metrics
            gpu_utilization.labels(gpu_id=gpu_id).set(util)
            gpu_memory_used.labels(gpu_id=gpu_id).set(mem_used)
            gpu_memory_total.labels(gpu_id=gpu_id).set(mem_total)
            gpu_temperature.labels(gpu_id=gpu_id).set(temp)
            gpu_power_usage.labels(gpu_id=gpu_id).set(power)
            
            logger.debug(
                f"GPU {gpu_id}: {util}% utilization, "
                f"{mem_used}/{mem_total}MB memory, {temp}°C"
            )
    
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timeout")
    except Exception as e:
        logger.warning(f"Error collecting GPU metrics: {e}")


def get_gpu_info() -> Optional[dict]:
    """Get GPU information for health checks."""
    if not check_gpu_available():
        return None
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "gpu_available": True,
                "gpu_name": parts[0].strip() if len(parts) > 0 else "Unknown",
                "driver_version": parts[1].strip() if len(parts) > 1 else "Unknown",
            }
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
    
    return {"gpu_available": False}

