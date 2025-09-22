#!/usr/bin/env python3
"""
Simple backend starter for the NLP to SQL prototype
"""

import subprocess
import sys
import os
from pathlib import Path

# Get the project root
PROJECT_ROOT = Path(__file__).parent

# Change to backend directory
backend_dir = PROJECT_ROOT / "backend"
os.chdir(backend_dir)

# Start uvicorn
cmd = [
    sys.executable,
    "-m",
    "uvicorn", 
    "app:app",
    "--reload",
    "--host", "0.0.0.0",
    "--port", "8000"
]

print("Starting NLP to SQL Backend Server...")
print(f"Working directory: {os.getcwd()}")
print(f"Command: {' '.join(cmd)}")
print("-" * 50)

# Run the server
subprocess.run(cmd)