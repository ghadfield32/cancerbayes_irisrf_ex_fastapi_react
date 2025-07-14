@echo off
REM Setup script for Python virtual environment (Windows)
REM ===================================================

echo 🔧 Setting up Python Virtual Environment
echo ========================================

REM Change to backend directory
cd backend

REM Check if virtual environment already exists
if exist "venv" (
    echo ✅ Virtual environment already exists
    echo 🔄 Activating existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo 📦 Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        echo 💡 Make sure Python is installed and accessible
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created successfully
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ All dependencies installed successfully

REM Test uvicorn installation
echo 🧪 Testing uvicorn installation...
python -c "import uvicorn; print('✅ uvicorn is available')"

if errorlevel 1 (
    echo ❌ uvicorn test failed
    pause
    exit /b 1
)

echo.
echo 🎉 Backend environment setup complete!
echo ===================================
echo 💡 To activate the environment manually:
echo    cd backend
echo    venv\Scripts\activate.bat
echo.
echo 🚀 To start the backend server:
echo    cd backend
echo    venv\Scripts\activate.bat
echo    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.

cd ..
pause 