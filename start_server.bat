@echo off
chcp 65001 >nul
echo ==========================================
echo   台风动画生成系统 - 启动脚本 (Windows)
echo ==========================================

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖
echo 检查依赖...
pip list | findstr /i "flask" >nul
if errorlevel 1 (
    echo 安装 Flask...
    pip install flask flask-cors
)

pip list | findstr /i "av " >nul
if errorlevel 1 (
    echo 安装 PyAV...
    pip install av
)

pip list | findstr /i "numpy" >nul
if errorlevel 1 (
    echo 安装 NumPy...
    pip install numpy
)

pip list | findstr /i "Pillow" >nul
if errorlevel 1 (
    echo 安装 Pillow...
    pip install Pillow
)

REM 创建必要目录
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

REM 检查必要文件
if not exist "animation.py" (
    echo 错误：未找到 animation.py
    pause
    exit /b 1
)

if not exist "config.py" (
    echo 错误：未找到 config.py
    pause
    exit /b 1
)

if not exist "api_server.py" (
    echo 错误：未找到 api_server.py
    pause
    exit /b 1
)

REM 检查素材目录
if not exist "tc_icons" (
    echo 警告：未找到 tc_icons 目录
)

if not exist "landfall_icons" (
    echo 警告：未找到 landfall_icons 目录
)

REM 启动 API 服务器
echo.
echo 启动 API 服务器...
echo 访问地址：http://localhost:5000
echo.
echo 按 Ctrl+C 停止服务器
echo ==========================================

python api_server.py

pause
