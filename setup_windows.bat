@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Biometric Voting System - Windows Setup
echo ============================================================
echo This project uses Python 3.10 x64 for MediaPipe compatibility.
echo.

py -3.10 --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.10 was not found through the Windows 'py' launcher.
  echo Install Python 3.10 x64 from https://www.python.org/downloads/release/python-31011/
  echo During installation, enable the Python Launcher option.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a dedicated .venv with Python 3.10...
  py -3.10 -m venv .venv
  if errorlevel 1 goto :failed
) else (
  echo Found existing .venv. Checking its Python version...
  .venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,10) else 1)"
  if errorlevel 1 (
    echo ERROR: Existing .venv is not Python 3.10.
    echo Rename or delete the project's .venv folder, then run setup_windows.bat again.
    pause
    exit /b 1
  )
)

echo.
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo.
echo Installing project requirements. This includes OpenCV-contrib and PyTorch and may take several minutes...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Checking imports and OpenCV face module...
.venv\Scripts\python.exe check_setup.py
if errorlevel 1 goto :failed

echo.
echo Setup checks passed.
echo Start the app with:
echo   .venv\Scripts\python.exe src\app.py
echo Then open http://127.0.0.1:5000/
echo.
echo NOTE: The ECAPA speaker model downloads the first time voice enrollment/verification is used.
echo Keep internet available until the model download completes.
pause
exit /b 0

:failed
echo.
echo Setup failed. Read the error above. If pip reports a version conflict, confirm Python 3.10 x64 is installed.
pause
exit /b 1
