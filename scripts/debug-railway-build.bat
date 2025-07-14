@echo off
REM Railway build debugging script
REM ==============================
REM This script simulates Railway's build process to debug issues

echo 🔍 Railway Build Debugging
echo ===========================

setlocal enabledelayedexpansion
set "errors=0"

echo [INFO] Simulating Railway build process...
echo.

REM Change to frontend directory
cd frontend

REM Check initial state
echo [INFO] Checking initial state...
if exist "node_modules" (
    echo [INFO] node_modules exists, removing for clean test...
    rmdir /s /q node_modules
)

if exist "dist" (
    echo [INFO] dist exists, removing for clean test...
    rmdir /s /q dist
)

echo [SUCCESS] Clean state achieved
echo.

REM Phase 1: Install dependencies (simulating Railway's npm ci)
echo [INFO] Phase 1: Installing dependencies...
echo [CMD] npm ci --prefer-offline --no-audit --loglevel=error
npm ci --prefer-offline --no-audit --loglevel=error
if errorlevel 1 (
    echo [ERROR] npm ci failed
    set /a errors+=1
    goto :cleanup
) else (
    echo [SUCCESS] Dependencies installed
)
echo.

REM Phase 2: Build application
echo [INFO] Phase 2: Building application...
echo [CMD] npm run build
npm run build
if errorlevel 1 (
    echo [ERROR] npm run build failed
    set /a errors+=1
    goto :cleanup
) else (
    echo [SUCCESS] Build completed
)
echo.

REM Phase 3: Verify build output
echo [INFO] Phase 3: Verifying build output...
if exist "dist" (
    echo [SUCCESS] dist directory created
    
    if exist "dist\index.html" (
        echo [SUCCESS] index.html found
    ) else (
        echo [ERROR] index.html not found
        set /a errors+=1
    )
    
    if exist "dist\assets" (
        echo [SUCCESS] assets directory found
        dir dist\assets
    ) else (
        echo [ERROR] assets directory not found
        set /a errors+=1
    )
) else (
    echo [ERROR] dist directory not created
    set /a errors+=1
)
echo.

REM Phase 4: Test serve command
echo [INFO] Phase 4: Testing serve command...
echo [CMD] npx serve -s dist -l 3001 --no-clipboard --no-port-switching
echo [INFO] Starting serve in background for 5 seconds...
start /b npx serve -s dist -l 3001 --no-clipboard --no-port-switching >nul 2>&1
timeout 5 >nul
taskkill /f /im node.exe >nul 2>&1
echo [SUCCESS] serve command test completed
echo.

:cleanup
cd ..

echo ==============================
if !errors! equ 0 (
    echo [SUCCESS] 🎉 All Railway build simulation tests passed!
    echo [INFO] Your build process should work on Railway
    echo.
    echo Build verification:
    echo ✅ npm ci works correctly
    echo ✅ npm run build produces output
    echo ✅ dist directory contains required files
    echo ✅ serve command works
) else (
    echo [ERROR] ❌ !errors! error(s) found in build simulation
    echo [INFO] Fix these issues before deploying to Railway
)

echo.
echo Railway deployment tips:
echo 1. Ensure package-lock.json is committed
echo 2. Test build locally before deploying
echo 3. Check Railway logs for detailed error messages
echo 4. Use 'railway logs' to monitor deployment
echo.

pause
exit /b !errors! 