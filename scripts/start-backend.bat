@echo off
REM Dedicated backend startup script with error handling
REM ==================================================

echo 🐍 Starting FastAPI Backend Server
echo ================================

REM Change to backend directory
cd /d "%~dp0..\backend"

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found
    echo 💡 Please run: scripts\setup-env.bat
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Set environment variables
set ENVIRONMENT=development
set PORT=8000
set PYTHONPATH=%CD%

REM Test if uvicorn is available
echo 🧪 Testing uvicorn availability...
python -c "import uvicorn; print('✅ uvicorn is available')"
if errorlevel 1 (
    echo ❌ uvicorn not available
    echo 💡 Please run: scripts\setup-env.bat
    pause
    exit /b 1
)

REM Test if the app can be imported
echo 🧪 Testing app import...
python -c "from app.main import app; print('✅ FastAPI app can be imported')"
if errorlevel 1 (
    echo ❌ Cannot import FastAPI app
    echo 💡 Check backend/app/main.py for errors
    pause
    exit /b 1
)

REM Start the server
echo 🚀 Starting FastAPI server on http://localhost:8000
echo 📚 API docs will be available at http://localhost:8000/api/docs
echo 🔄 Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --log-level info

echo.
echo 👋 Backend server stopped
pause 