@echo off
REM API服务启动脚本 (Windows版本)

setlocal enabledelayedexpansion

echo ==========================================
echo LearningFriend API 服务启动脚本
echo ==========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python未安装
    exit /b 1
)

echo.
echo Python版本:
python --version

REM 安装依赖
echo.
echo 检查并安装依赖...
pip install -r requirements-api.txt
if errorlevel 1 (
    echo 错误: 依赖安装失败
    exit /b 1
)

REM 检查配置文件
echo.
if not exist "config\config.yaml" (
    echo 警告: 配置文件不存在: config\config.yaml
    echo 请复制 config\config.yaml.example 并配置相关参数
    exit /b 1
)

echo ✓ 配置文件检查通过

REM 选择启动模式
echo.
echo ==========================================
echo 选择启动模式:
echo   1. 主服务 (端口 8000) - 推荐
echo   2. ASR服务 (端口 8001)
echo   3. LLM服务 (端口 8002)
echo   4. 全部服务 (8000, 8001, 8002)
echo ==========================================
echo.

set /p choice="请输入选项 (1-4): "

if "%choice%"=="1" (
    echo.
    echo 启动主服务...
    python -m src.api.main
) else if "%choice%"=="2" (
    echo.
    echo 启动ASR服务...
    python -m src.api.asr_api
) else if "%choice%"=="3" (
    echo.
    echo 启动LLM服务...
    python -m src.api.llm_api
) else if "%choice%"=="4" (
    echo.
    echo 启动所有服务...
    
    REM 创建日志目录
    if not exist "logs" mkdir logs
    
    REM 后台启动ASR和LLM服务
    start /B python -m src.api.asr_api > logs\asr_api.log 2>&1
    echo ASR服务已启动 (端口: 8001)
    
    start /B python -m src.api.llm_api > logs\llm_api.log 2>&1
    echo LLM服务已启动 (端口: 8002)
    
    REM 等待一下让服务启动
    timeout /t 2 /nobreak >nul
    
    REM 前台运行主服务
    echo 主服务启动中 (端口: 8000)...
    python -m src.api.main
) else (
    echo 无效选项
    exit /b 1
)

endlocal
