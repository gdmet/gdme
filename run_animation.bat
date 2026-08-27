@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
  echo [ERROR] Project environment is not initialized.
  echo Double-click env_initialize.bat first.
  pause
  exit /b 1
)

"%VENV_PYTHON%" "%PROJECT_DIR%animation.py" %*

if errorlevel 1 pause
endlocal
