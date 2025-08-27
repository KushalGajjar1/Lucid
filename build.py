#!/usr/bin/env python3
"""
Build script for Lucid Tokenizer with Rust extension
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_rust():
    """Check if Rust is installed"""
    print("🔍 Checking Rust installation...")
    
    if not shutil.which("cargo"):
        print("❌ Rust/Cargo not found!")
        print("Please install Rust from https://rustup.rs/")
        return False
    
    try:
        result = subprocess.run(["cargo", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ Rust found: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to run cargo --version")
        return False

def check_python_deps():
    """Check and install Python dependencies"""
    print("🔍 Checking Python dependencies...")
    
    required_packages = ["setuptools", "wheel", "setuptools-rust"]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is available")
        except ImportError:
            print(f"📦 Installing {package}...")
            if not run_command(f"pip install {package}", f"Installing {package}"):
                return False
    
    return True

def build_rust():
    """Build the Rust extension"""
    print("🔨 Building Rust extension...")
    
    # Clean previous builds
    if os.path.exists("target"):
        print("🧹 Cleaning previous Rust builds...")
        shutil.rmtree("target")
    
    # Build in release mode
    if not run_command("cargo build --release", "Building Rust extension"):
        return False
    
    return True

def install_package():
    """Install the package in development mode"""
    print("📦 Installing package in development mode...")
    
    if not run_command("pip install -e .", "Installing package"):
        return False
    
    return True

def test_package():
    """Test the installed package"""
    print("🧪 Testing the installed package...")
    
    if not run_command("python test_tokenizer.py", "Running tests"):
        return False
    
    return True

def main():
    """Main build process"""
    print("🚀 Building Lucid Tokenizer with Rust Extension")
    print("=" * 60)
    
    # Check prerequisites
    if not check_rust():
        sys.exit(1)
    
    if not check_python_deps():
        sys.exit(1)
    
    # Build process
    if not build_rust():
        print("❌ Rust build failed!")
        sys.exit(1)
    
    if not install_package():
        print("❌ Package installation failed!")
        sys.exit(1)
    
    if not test_package():
        print("❌ Package testing failed!")
        sys.exit(1)
    
    print("\n🎉 Build completed successfully!")
    print("You can now use the tokenizer:")
    print("  from Lucid.Tokenizer import BPETokenizer")
    print("  tokenizer = BPETokenizer()")

if __name__ == "__main__":
    main()
