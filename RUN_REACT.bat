@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Vova Screener React

echo.
echo  ========================================
echo   Vova Screener React - starting...
echo  ========================================
echo.

REM Require React stack (branch cursor/react-mongo-refactor-design-18b3)
if not exist "package.json" goto :missing_react
if not exist "apps\web\package.json" goto :missing_react

where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm not found. Install Node.js 20+ from https://nodejs.org
    pause
    exit /b 1
)

echo Stopping previous React/API processes on ports 5173 and 3001...
call :kill_port 5173
call :kill_port 3001
taskkill /IM esbuild.exe /F >nul 2>&1

REM concurrently alone is not enough — leftover root deps can skip a full workspace install
set "NEED_INSTALL=0"
if not exist "node_modules\concurrently" set "NEED_INSTALL=1"
if not exist "node_modules\vite" set "NEED_INSTALL=1"
if not exist "node_modules\ts-node" set "NEED_INSTALL=1"
if not exist "node_modules\@vova\web" set "NEED_INSTALL=1"
if not exist "node_modules\@vova\api" set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="1" (
    echo Installing npm dependencies...
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed.
        pause
        exit /b 1
    )
)

echo Opening browser at http://localhost:5173
echo Close this window to stop the React app.
echo.
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:5173"

call npm run dev

if errorlevel 1 (
    echo.
    echo React app exited with an error.
    pause
    exit /b 1
)

endlocal
exit /b 0

:missing_react
echo ERROR: React app files not found in this folder.
echo.
echo Switch to the React branch first, then run this bat again:
echo   git checkout cursor/react-mongo-refactor-design-18b3
echo.
pause
exit /b 1

:kill_port
set "PORT=%~1"
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":%PORT% " ^| findstr LISTENING') do (
    if not "%%P"=="0" taskkill /PID %%P /F >nul 2>&1
)
exit /b 0
