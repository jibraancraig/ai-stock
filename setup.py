#!/usr/bin/env python3
"""
Setup Script for Stock Market AI
Installs dependencies and initializes the project
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Stock Market AI - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return 1
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return 1
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = ["data", "data/models", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import sklearn
        import tensorflow as tf
        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return 1
    
    # Test data collection
    print("\n📊 Testing data collection...")
    try:
        from data_collector import StockDataCollector
        collector = StockDataCollector()
        data = collector.fetch_stock_data("AAPL", period="1mo")
        print(f"✅ Data collection test successful - collected {len(data)} records")
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the demo: python quick_start.py")
    print("2. Train models: python train_models.py --symbol AAPL")
    print("3. Launch dashboard: python app.py")
    print("4. Get predictions: python predict.py --symbol AAPL")
    
    return 0

if __name__ == "__main__":
    exit(main()) 