@echo off
"%~dp0..\build\sdvk-server.exe" --spv "%~dp0..\build\shaders" --web "%~dp0..\webui" --out-dir "%~dp0..\output" %*
