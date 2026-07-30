@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Vova Home Server
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\home-server\start-home-server.ps1" -Detach
if errorlevel 1 (
  echo.
  echo Home server failed to start.
  pause
  exit /b 1
)
endlocal
