@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Vova Cloudflare Tunnel
echo.
echo  Starts Cloudflare Tunnel to http://127.0.0.1:5173
echo  Use Quick URL on your phone with MOBILE DATA to verify "from anywhere".
echo  Home server must already be running (RUN_HOME_SERVER.bat).
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\home-server\start-tunnel.ps1" -Quick
if errorlevel 1 (
  echo.
  echo Tunnel exited with an error.
  pause
  exit /b 1
)
endlocal
