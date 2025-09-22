#!/usr/bin/env python3
"""
Launch script for NLP to SQL prototype
Starts both backend API and opens frontend in browser
"""

import subprocess
import webbrowser
import time
import os
import sys
import threading
import socket
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def is_port_open(host, port):
    """Check if a port is open"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting backend API server...")
    
    # Check if port 8000 is already in use
    if is_port_open('localhost', 8000):
        print("⚠️  Port 8000 is already in use. Please stop the existing server.")
        return False
    
    # Start the backend server
    backend_dir = PROJECT_ROOT / "backend"
    cmd = [sys.executable, "-m", "uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor startup
        for line in iter(process.stdout.readline, ''):
            print(f"Backend: {line.strip()}")
            if "Uvicorn running on" in line:
                print("✅ Backend server is running!")
                return True
            if "error" in line.lower():
                print("❌ Error starting backend server")
                return False
                
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False
    
    return True

def open_frontend():
    """Open the frontend in the default web browser"""
    print("🌐 Opening frontend in browser...")
    
    frontend_path = PROJECT_ROOT / "frontend" / "index.html"
    
    if not frontend_path.exists():
        print("❌ Frontend file not found!")
        return False
    
    # Wait a bit for the backend to be fully ready
    time.sleep(2)
    
    # Open in browser
    file_url = f"file:///{frontend_path.absolute()}"
    webbrowser.open(file_url)
    print("✅ Frontend opened in browser!")
    
    return True

def main():
    """Main launcher function"""
    print("\n" + "="*60)
    print("NLP to SQL Prototype Launcher")
    print("="*60 + "\n")
    
    # Check dependencies
    try:
        import fastapi
        import uvicorn
        import anthropic
        print("✅ All required dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install dependencies with:")
        print("pip install -r requirements.txt")
        return
    
    # Check if .env file exists
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found. API will run in mock mode.")
    else:
        from config.settings import get_settings
        settings = get_settings()
        if settings.claude_api_key and settings.claude_api_key != "test_api_key":
            print("✅ Claude API key configured")
        else:
            print("⚠️  Warning: No valid Claude API key. Running in mock mode.")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to start
    print("\nWaiting for backend to start...")
    for i in range(30):  # Wait up to 30 seconds
        if is_port_open('localhost', 8000):
            print("✅ Backend is ready!")
            break
        time.sleep(1)
    else:
        print("❌ Backend failed to start within 30 seconds")
        return
    
    # Open frontend
    open_frontend()
    
    print("\n" + "="*60)
    print("🎉 NLP to SQL Prototype is running!")
    print("="*60)
    print("\nBackend API: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Frontend: Check your browser")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down NLP to SQL prototype...")

if __name__ == "__main__":
    main()