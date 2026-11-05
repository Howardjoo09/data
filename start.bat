@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    echo Python 3.11 or newer is required. Install from https://www.python.org/downloads/windows/
    pause
    exit /b 1
)
py -3 app.py
if errorlevel 1 pause

