#!/usr/bin/env python
"""
Build script for Render deployment
"""
import os
import subprocess
import sys

def run_command(command):
    """Run a command and handle errors"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

def main():
    """Main build process"""
    print("Starting build process...")
    
    # Install dependencies
    run_command("pip install --upgrade pip")
    run_command("pip install -r requirements.txt")
    
    # Collect static files
    run_command("python manage.py collectstatic --noinput")
    
    # Run migrations
    run_command("python manage.py migrate")
    
    print("Build completed successfully!")

if __name__ == "__main__":
    main()
