#!/bin/bash

echo "==========================================="
echo "   DGA Domain Detector - Quick Start"
echo "==========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

echo ""
echo "==========================================="
echo "   Installation Complete!"
echo "==========================================="
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Train the model:"
echo "   jupyter notebook dga_detection_training.ipynb"
echo ""
echo "2. Run the web interface:"
echo "   streamlit run main.py"
echo ""
echo "==========================================="
