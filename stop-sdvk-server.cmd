@echo off
setlocal
rem Ask the running server to release VRAM first, then stop it.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Method POST http://127.0.0.1:7500/api/unload | Out-Null } catch {}"
taskkill /IM sdvk-server.exe /F >nul 2>&1
echo sdvk-server stopped.
