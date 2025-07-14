@echo off
REM Development script for FastAPI + React application (Windows)
REM ============================================================

echo 🚀 Starting FastAPI + React Development Environment
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ All dependencies are available

REM Install dependencies if --install argument is provided
if "%1"=="--install" (
    echo 📦 Installing dependencies...
    
    REM Install backend dependencies
    echo Installing backend dependencies...
    cd backend
    if not exist venv (
        python -m venv venv
    )
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    cd ..
    
    REM Install frontend dependencies
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
    
    echo ✅ All dependencies installed
)

REM Start backend server
echo 🐍 Starting FastAPI backend on port 8000...
cd backend
if not exist venv (
    echo ❌ Virtual environment not found. Please run setup first.
    echo 💡 Run: scripts\setup-env.bat
    pause
    exit /b 1
)
set ENVIRONMENT=development
start "FastAPI Backend" cmd /k "venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
cd ..

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server
echo ⚛️ Starting React frontend on port 5173...
cd frontend
set VITE_API_URL=http://localhost:8000
start "React Frontend" cmd /k "npm run dev"
cd ..

echo.
echo 🎉 Development environment is running!
echo ==================================
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/api/docs
echo.
echo Press any key to continue (servers will keep running in separate windows)
pause >nul 