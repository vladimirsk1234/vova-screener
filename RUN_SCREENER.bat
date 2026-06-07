@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Vova Screener

echo.
echo  ========================================
echo   Vova Screener - starting...
echo  ========================================
echo.

REM Full local RAM (not Streamlit Cloud limits)
set SCREENER_LOW_MEMORY=0

REM Use project virtualenv when present
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

REM Pick Python: venv, py launcher, or python on PATH
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
    echo ERROR: Python not found. Install Python 3.10+ from https://python.org
    echo        Or create a venv:  py -3 -m venv .venv
    pause
    exit /b 1
)

REM Install deps once if streamlit missing
%PY% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies from requirements.txt ...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: pip install failed.
        pause
        exit /b 1
    )
)

echo Opening browser at http://localhost:8501
echo Close this window to stop the screener.
echo.

%PY% -m streamlit run headless_scanner.py --server.headless false

if errorlevel 1 (
    echo.
    echo Screener exited with an error.
    pause
    exit /b 1
)

endlocal
