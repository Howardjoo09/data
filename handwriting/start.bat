@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer is required. Install from https://www.python.org/downloads/windows/
    pause
    exit /b 1
)
if not exist .venv\Scripts\python.exe (
    py -3 -m venv .venv
    if errorlevel 1 goto fail
)
.venv\Scripts\python -m pip install -q -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python server.py --open
if errorlevel 1 goto fail
exit /b 0
:fail
echo Setup failed. Check the messages above.
pause
exit /b 1
