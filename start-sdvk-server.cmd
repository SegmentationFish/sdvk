@echo off
setlocal
cd /d "%~dp0"

rem One instance only: a second copy would fail to bind port 7500 and close instantly.
netstat -ano | findstr ":7500" | findstr LISTENING >nul
if not errorlevel 1 (
  echo sdvk-server is already running: http://127.0.0.1:7500/
  start "" "http://127.0.0.1:7500/"
  exit /b 0
)

rem Detached, hidden process: closing this window or the agent that started it will not kill it.
powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '%~dp0build\sdvk-server.exe' -ArgumentList '--port 7500 --lan --spv shaders --out-dir output --web ..\webui --idle-unload 10' -WorkingDirectory '%~dp0build' -RedirectStandardOutput '%~dp0server.log' -RedirectStandardError '%~dp0server.err.log' -WindowStyle Hidden"

timeout /t 3 /nobreak >nul
netstat -ano | findstr ":7500" | findstr LISTENING >nul
if errorlevel 1 (
  echo sdvk-server failed to start. See server.log / server.err.log
  pause
  exit /b 1
)

echo sdvk-server running: http://127.0.0.1:7500/
echo Idle models are unloaded after 10 minutes to release VRAM.
start "" "http://127.0.0.1:7500/"
