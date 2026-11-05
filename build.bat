@echo off
setlocal
cd /d "%~dp0"
if not exist .build-venv\Scripts\python.exe (
    py -3 -m venv .build-venv
    if errorlevel 1 goto :failed
)
.build-venv\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 goto :failed
.build-venv\Scripts\python.exe -m unittest discover -s tests -v
if errorlevel 1 goto :failed
.build-venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed --name HangulStepper app.py
if errorlevel 1 goto :failed
echo Created: dist\HangulStepper.exe
pause
exit /b 0
:failed
echo Build failed. See the error above.
pause
exit /b 1
