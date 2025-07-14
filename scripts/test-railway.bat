@echo off
REM Railway deployment testing script for Windows
REM ============================================

echo 🚂 Testing Railway Deployment Setup
echo ===================================

REM Check if Railway CLI is installed
echo [INFO] Checking Railway CLI installation...
railway --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Railway CLI is not installed
    echo Install it with: npm install -g @railway/cli
    pause
    exit /b 1
)
echo [SUCCESS] Railway CLI is installed

REM Check if user is logged in
echo [INFO] Checking Railway authentication...
railway whoami >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not logged in to Railway
    echo Login with: railway login
    pause
    exit /b 1
)
echo [SUCCESS] Railway authentication verified

REM Test backend configuration
echo [INFO] Testing backend configuration...
if exist "backend\railway.json" (
    echo [SUCCESS] Backend railway.json found
    
    REM Check for uvicorn command
    findstr /C:"uvicorn app.main:app" backend\railway.json >nul
    if errorlevel 1 (
        echo [ERROR] Backend start command may be incorrect
        pause
        exit /b 1
    )
    echo [SUCCESS] Backend start command is correct
) else (
    echo [ERROR] Backend railway.json not found
    pause
    exit /b 1
)

REM Test frontend configuration
echo [INFO] Testing frontend configuration...
if exist "frontend\railway.json" (
    echo [SUCCESS] Frontend railway.json found
    
    REM Check for serve command
    findstr /C:"serve" frontend\railway.json >nul
    if errorlevel 1 (
        echo [ERROR] Frontend serve command not found
        pause
        exit /b 1
    )
    echo [SUCCESS] Frontend serve command found
) else (
    echo [ERROR] Frontend railway.json not found
    pause
    exit /b 1
)

REM Test backend locally
echo [INFO] Testing backend locally...
cd backend
if not exist "venv" (
    echo [WARNING] Virtual environment not found, creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

python -c "from app.main import app; print('✅ FastAPI app imports successfully')" 2>nul
if errorlevel 1 (
    echo [ERROR] Backend app import failed
    cd ..
    pause
    exit /b 1
)
echo [SUCCESS] Backend app imports successfully
cd ..

REM Test frontend build
echo [INFO] Testing frontend build...
cd frontend
if not exist "node_modules" (
    echo [WARNING] Node modules not found, installing...
    npm ci
)

npm run build >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Frontend build failed
    cd ..
    pause
    exit /b 1
)
echo [SUCCESS] Frontend builds successfully

if exist "dist" (
    echo [SUCCESS] Frontend dist directory created
) else (
    echo [ERROR] Frontend dist directory not found after build
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo [SUCCESS] 🎉 All tests completed!
echo.
echo Next steps for Railway deployment:
echo 1. Create two services in Railway:
echo    - Backend service with root directory: backend/
echo    - Frontend service with root directory: frontend/
echo 2. Set environment variables:
echo    - Backend: PORT (automatically set by Railway)
echo    - Frontend: Set VITE_API_URL to your backend service URL
echo 3. Deploy both services
echo.
echo 🔗 Useful Railway commands:
echo    railway link    # Link to Railway project
echo    railway up      # Deploy current directory
echo    railway logs    # View deployment logs
echo    railway shell   # Open shell with Railway environment
echo.

pause 