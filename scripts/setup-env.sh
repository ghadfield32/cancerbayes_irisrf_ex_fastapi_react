#!/bin/bash
# Setup script for Python virtual environment (Unix/Linux/macOS)
# ============================================================

set -e

echo "🔧 Setting up Python Virtual Environment"
echo "========================================"

# Change to backend directory
cd backend

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment already exists"
    echo "🔄 Activating existing virtual environment..."
    source venv/bin/activate
else
    echo "📦 Creating new virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "💡 Make sure Python 3 is installed and accessible"
        exit 1
    fi
    echo "✅ Virtual environment created successfully"
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Upgrade pip
echo "🔄 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ All dependencies installed successfully"

# Test uvicorn installation
echo "🧪 Testing uvicorn installation..."
python -c "import uvicorn; print('✅ uvicorn is available')"

if [ $? -ne 0 ]; then
    echo "❌ uvicorn test failed"
    exit 1
fi

echo ""
echo "🎉 Backend environment setup complete!"
echo "==================================="
echo "💡 To activate the environment manually:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo ""
echo "🚀 To start the backend server:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""

cd .. 