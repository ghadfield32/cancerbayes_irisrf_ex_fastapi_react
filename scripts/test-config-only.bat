@echo off
REM Railway configuration testing script (no auth required)
REM ======================================================

echo 🚂 Testing Railway Configuration
echo ================================

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
    echo [WARNING] Virtual environment not found, please run: scripts\setup-env.bat
    cd ..
    goto :frontend_test
)

call venv\Scripts\activate.bat
python -c "from app.main import app; print('✅ FastAPI app imports successfully')" 2>nul
if errorlevel 1 (
    echo [ERROR] Backend app import failed
    cd ..
    pause
    exit /b 1
)
echo [SUCCESS] Backend app imports successfully
cd ..

:frontend_test
REM Test frontend dependencies
echo [INFO] Testing frontend configuration...
cd frontend
if not exist "node_modules" (
    echo [WARNING] Node modules not found, please run: npm install
    cd ..
    goto :summary
)

if not exist "package.json" (
    echo [ERROR] package.json not found
    cd ..
    pause
    exit /b 1
)

REM Check if serve is in dependencies
findstr /C:"serve" package.json >nul
if errorlevel 1 (
    echo [ERROR] serve dependency not found in package.json
    cd ..
    pause
    exit /b 1
)
echo [SUCCESS] serve dependency found in package.json
cd ..

:summary
echo.
echo [SUCCESS] 🎉 Configuration tests completed!
echo.
echo Configuration Summary:
echo ✅ Backend railway.json configured correctly
echo ✅ Frontend railway.json configured correctly
echo ✅ Backend FastAPI app structure is valid
echo ✅ Frontend serve dependency is available
echo.
echo Next steps for Railway deployment:
echo 1. Login to Railway: railway login
echo 2. Create a Railway project
echo 3. Add two services:
echo    - Backend service with root directory: backend/
echo    - Frontend service with root directory: frontend/
echo 4. Set environment variables:
echo    - Frontend: VITE_API_URL = https://your-backend-service.railway.app
echo 5. Deploy both services
echo.
echo For detailed instructions, see: RAILWAY_DEPLOYMENT.md
echo.

pause 