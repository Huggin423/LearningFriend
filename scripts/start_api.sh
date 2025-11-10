#!/bin/bash
# API服务启动脚本

set -e

echo "=========================================="
echo "LearningFriend API 服务启动脚本"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: Python未安装"
    exit 1
fi

echo ""
echo "Python版本:"
python --version

# 安装依赖
echo ""
echo "检查并安装依赖..."
pip install -r requirements-api.txt

# 检查配置文件
echo ""
if [ ! -f "config/config.yaml" ]; then
    echo "警告: 配置文件不存在: config/config.yaml"
    echo "请复制 config/config.yaml.example 并配置相关参数"
    exit 1
fi

echo "✓ 配置文件检查通过"

# 选择启动模式
echo ""
echo "=========================================="
echo "选择启动模式:"
echo "  1. 主服务 (端口 8000) - 推荐"
echo "  2. ASR服务 (端口 8001)"
echo "  3. LLM服务 (端口 8002)"
echo "  4. 全部服务 (8000, 8001, 8002)"
echo "=========================================="

read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "启动主服务..."
        python -m src.api.main
        ;;
    2)
        echo ""
        echo "启动ASR服务..."
        python -m src.api.asr_api
        ;;
    3)
        echo ""
        echo "启动LLM服务..."
        python -m src.api.llm_api
        ;;
    4)
        echo ""
        echo "启动所有服务..."
        echo "主服务将在后台运行..."
        
        # 后台启动ASR和LLM服务
        nohup python -m src.api.asr_api > logs/asr_api.log 2>&1 &
        ASR_PID=$!
        echo "ASR服务已启动 (PID: $ASR_PID, 端口: 8001)"
        
        nohup python -m src.api.llm_api > logs/llm_api.log 2>&1 &
        LLM_PID=$!
        echo "LLM服务已启动 (PID: $LLM_PID, 端口: 8002)"
        
        # 前台运行主服务
        echo "主服务启动中 (端口: 8000)..."
        python -m src.api.main
        
        # 主服务退出后，停止其他服务
        echo "停止所有服务..."
        kill $ASR_PID $LLM_PID 2>/dev/null || true
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac
