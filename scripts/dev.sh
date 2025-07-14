#!/bin/bash

# Development script for FastAPI + React application
# ==================================================

set -e

echo "🚀 Starting FastAPI + React Development Environment"
echo "=================================================="

# Check if required dependencies are installed
check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js is not installed"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        echo "❌ npm is not installed"
        exit 1
    fi
    
    echo "✅ All dependencies are available"
}

# Install backend dependencies
install_backend() {
    echo "📦 Installing backend dependencies..."
    cd backend
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
    echo "✅ Backend dependencies installed"
}

# Install frontend dependencies
install_frontend() {
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    echo "✅ Frontend dependencies installed"
}

# Start backend server
start_backend() {
    echo "🐍 Starting FastAPI backend on port 8000..."
    cd backend
    if [ ! -d "venv" ]; then
        echo "❌ Virtual environment not found. Please run setup first."
        echo "💡 Run: ./scripts/setup-env.sh"
        exit 1
    fi
    source venv/bin/activate
    export ENVIRONMENT=development
    python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 &
    BACKEND_PID=$!
    cd ..
    echo "✅ Backend started (PID: $BACKEND_PID)"
}

# Start frontend server
start_frontend() {
    echo "⚛️  Starting React frontend on port 5173..."
    cd frontend
    export VITE_API_URL=http://localhost:8000
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    echo "✅ Frontend started (PID: $FRONTEND_PID)"
}

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo "👋 Development servers stopped"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    check_dependencies
    
    # Install dependencies if --install flag is provided
    if [[ "$1" == "--install" ]]; then
        install_backend
        install_frontend
    fi
    
    start_backend
    sleep 3  # Give backend time to start
    start_frontend
    
    echo ""
    echo "🎉 Development environment is running!"
    echo "=================================="
    echo "📱 Frontend: http://localhost:5173"
    echo "🔧 Backend API: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/api/docs"
    echo ""
    echo "Press Ctrl+C to stop all servers"
    
    # Wait for user to stop
    wait
}

# Run main function
main "$@" 