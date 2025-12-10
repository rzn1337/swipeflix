#!/usr/bin/env python3
"""Start Evidently service locally."""

import subprocess
import sys
from pathlib import Path


def start_evidently():
    """Start Evidently service on port 7000."""
    evidently_path = Path("monitoring/evidently")
    evidently_path.mkdir(parents=True, exist_ok=True)

    try:
        # Try to import evidently (check if available)
        import importlib.util

        spec = importlib.util.find_spec("evidently")
        if spec is None:
            raise ImportError("evidently not installed")

        print("Starting Evidently service on http://localhost:7000")
        print(f"Workspace: {evidently_path.absolute()}")

        # Start Evidently UI
        subprocess.run(
            [
                sys.executable,
                "-m",
                "evidently",
                "ui",
                "--host",
                "0.0.0.0",
                "--port",
                "7000",
                "--workspace",
                str(evidently_path.absolute()),
            ]
        )
    except ImportError:
        print("Evidently not installed.")
        print("Install with: pip install evidently")
        print("Or use the web service at: https://evidentlyai.com")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvidently service stopped.")
        sys.exit(0)


if __name__ == "__main__":
    start_evidently()
