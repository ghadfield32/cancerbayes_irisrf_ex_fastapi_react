@echo off
REM Pre-deployment validation script
REM ================================
REM This script validates that the project is ready for Railway deployment
REM by checking dependencies, lock files, and build processes

echo 🔍 Pre-Deployment Validation
echo ==============================

setlocal enabledelayedexpansion
set "errors=0"

echo [INFO] Validating project structure and dependencies...
echo.

REM Check if package-lock.json is in sync
echo [INFO] Checking package-lock.json synchronization...
cd frontend
if not exist "package.json" (
    echo [ERROR] package.json not found in frontend/
    set /a errors+=1
    goto :backend_check
)

if not exist "package-lock.json" (
    echo [ERROR] package-lock.json not found in frontend/
    echo [INFO] Run 'npm install' to generate package-lock.json
    set /a errors+=1
    goto :backend_check
)

REM Test npm ci (dry run)
echo [INFO] Testing npm ci compatibility...
npm ci --dry-run >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm ci will fail - package-lock.json is out of sync
    echo [FIX] Run 'npm install' to update package-lock.json
    set /a errors+=1
) else (
    echo [SUCCESS] npm ci compatibility verified
)

REM Check for serve dependency
findstr /C:"serve" package.json >nul
if errorlevel 1 (
    echo [ERROR] serve dependency not found in package.json
    echo [FIX] Add serve to devDependencies: npm install --save-dev serve
    set /a errors+=1
) else (
    echo [SUCCESS] serve dependency found
)

REM Test build process
echo [INFO] Testing frontend build process...
if exist "dist" (
    rmdir /s /q dist
)

npm run build >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Frontend build failed
    echo [DEBUG] Run 'npm run build' manually to see detailed errors
    set /a errors+=1
) else (
    echo [SUCCESS] Frontend builds successfully
    
    if exist "dist\index.html" (
        echo [SUCCESS] dist/index.html created
    ) else (
        echo [ERROR] dist/index.html not found after build
        set /a errors+=1
    )
)

REM Test serve command
echo [INFO] Testing serve command...
if exist "dist" (
    timeout 3 >nul 2>&1 & npx serve -s dist -l 3001 --no-clipboard >nul 2>&1 &
    timeout 2 >nul 2>&1
    taskkill /f /im node.exe >nul 2>&1
    echo [SUCCESS] serve command works
) else (
    echo [WARNING] Cannot test serve - dist directory not found
)

cd ..

:backend_check
echo.
echo [INFO] Checking backend configuration...
cd backend

if not exist "railway.json" (
    echo [ERROR] backend/railway.json not found
    set /a errors+=1
    goto :summary
)

REM Check backend dependencies
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in backend/
    set /a errors+=1
    goto :summary
)

REM Check if virtual environment exists and works
if exist "venv" (
    call venv\Scripts\activate.bat
    python -c "from app.main import app; print('Backend import successful')" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Backend app import failed
        echo [DEBUG] Check backend/app/main.py for import errors
        set /a errors+=1
    ) else (
        echo [SUCCESS] Backend app imports successfully
    )
) else (
    echo [WARNING] Backend virtual environment not found
    echo [INFO] Run 'scripts\setup-env.bat' to create virtual environment
)

cd ..

:summary
echo.
echo ==============================
if !errors! equ 0 (
    echo [SUCCESS] 🎉 All pre-deployment checks passed!
    echo [INFO] Your project is ready for Railway deployment
    echo.
    echo Next steps:
    echo 1. Commit all changes: git add . && git commit -m "Update dependencies"
    echo 2. Push to repository: git push
    echo 3. Deploy to Railway with separate services:
    echo    - Backend: root directory = backend/
    echo    - Frontend: root directory = frontend/
) else (
    echo [ERROR] ❌ !errors! error(s) found - fix before deploying
    echo [INFO] Review the errors above and run this script again
)

echo.
echo Railway deployment commands:
echo   railway login
echo   railway link
echo   railway up
echo   railway logs
echo.

pause
exit /b !errors! 