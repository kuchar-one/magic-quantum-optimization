#!/bin/bash

# Magic Quantum Sequence Optimization - Installation Script
# This script sets up the environment and installs dependencies

set -e  # Exit on any error

echo "🔮 Magic Quantum Sequence Optimization - Installation"
echo "=================================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ is required. Found: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✅ Pip upgraded"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .
echo "✅ Package installed"

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import qutip; print(f'QuTiP version: {qutip.__version__}')"
python -c "import pymoo; print(f'PyMOO version: {pymoo.__version__}')"
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
echo "✅ Installation verified"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run the web interface: python run_app.py"
echo "3. Or run optimization: python quo.py --help"
echo ""
echo "For more information, see README.md"
