@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title Tropical Cyclone Season Animator - Environment Initializer

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_EXE="

echo ============================================================
echo Tropical Cyclone Season Animator environment initialization
echo Project: %PROJECT_DIR%
echo ============================================================
echo.

if exist "%VENV_PYTHON%" goto install_dependencies

echo [1/4] Searching for a usable 64-bit Python 3.10 or newer...
call :find_python

if not defined PYTHON_EXE (
  echo No suitable Python was found. Installing Python 3.12 for the current user...
  where winget >nul 2>nul
  if not errorlevel 1 winget install --id Python.Python.3.12 --exact --scope user --silent --accept-package-agreements --accept-source-agreements
  call :find_python
)

if not defined PYTHON_EXE (
  echo winget was unavailable or unsuccessful. Downloading the official installer...
  set "PYTHON_INSTALLER=%TEMP%\typhoon-python-3.12.10-amd64.exe"
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $env:PYTHON_INSTALLER"
  if errorlevel 1 goto python_install_failed
  start "" /wait "!PYTHON_INSTALLER!" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=1 Include_test=0 Shortcuts=0
  set "INSTALL_RESULT=!ERRORLEVEL!"
  del /q "!PYTHON_INSTALLER!" >nul 2>nul
  if not "!INSTALL_RESULT!"=="0" goto python_install_failed
  call :find_python
)

if not defined PYTHON_EXE goto python_install_failed

echo [2/4] Creating isolated environment: .venv
"%PYTHON_EXE%" -m venv "%VENV_DIR%"
if errorlevel 1 goto venv_failed

:install_dependencies
echo [3/4] Installing or updating Python dependencies...
set "PIP_PREFER_BINARY=1"
"%VENV_PYTHON%" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if errorlevel 1 goto dependency_failed
"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 goto dependency_failed

echo [4/4] Verifying the environment...
"%VENV_PYTHON%" -c "import struct,sys,tkinter,av,cv2,numpy,PIL; assert sys.version_info >= (3,10); assert struct.calcsize('P') * 8 == 64; print('Python', sys.version.split()[0]); print('PyAV', av.__version__); print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__); print('Pillow', PIL.__version__)"
if errorlevel 1 goto verification_failed

echo.
echo ============================================================
echo Environment initialized successfully.
echo You can now run run_animation.bat or preview_config.py.
echo ============================================================
pause
exit /b 0

:python_install_failed
echo.
echo [ERROR] Python 3.12 installation failed.
goto failed

:venv_failed
echo.
echo [ERROR] Failed to create .venv.
goto failed

:dependency_failed
echo.
echo [ERROR] Failed to install dependencies. Check the network connection and retry.
goto failed

:verification_failed
echo.
echo [ERROR] Dependencies were installed, but the verification step failed.
goto failed

:failed
echo Initialization was not completed.
pause
exit /b 1

:find_python
set "PYTHON_EXE="
for /f "usebackq delims=" %%P in (`py -3.12 -c "import struct,sys; sys.exit(1) if struct.calcsize('P') * 8 != 64 else print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE for /f "usebackq delims=" %%P in (`python -c "import struct,sys; sys.exit(1) if sys.version_info ^< (3,10) or struct.calcsize('P') ^* 8 != 64 else print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
exit /b 0
