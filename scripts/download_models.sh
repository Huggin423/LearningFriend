#!/bin/bash
# 模型下载脚本

echo "======================================"
echo "智能学伴系统 - 模型下载脚本"
echo "======================================"

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

echo -e "${YELLOW}项目目录: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}模型目录: $MODELS_DIR${NC}"
echo ""

# 创建模型目录
mkdir -p "$MODELS_DIR/funasr"
mkdir -p "$MODELS_DIR/indextts2"

# FunASR模型会自动下载，无需手动处理
echo -e "${GREEN}[1/2] FunASR模型${NC}"
echo "FunASR模型会在首次运行时自动从ModelScope下载"
echo "模型名称: paraformer-zh"
echo "下载位置: ~/.cache/modelscope/"
echo ""

# IndexTTS2模型下载
echo -e "${GREEN}[2/2] IndexTTS2模型${NC}"
echo -e "${YELLOW}注意: IndexTTS2模型需要手动下载${NC}"
echo ""
echo "IndexTTS2模型下载步骤："
echo "1. 访问 IndexTTS2 官方仓库或模型托管平台"
echo "2. 下载预训练模型文件："
echo "   - config.json"
echo "   - checkpoint.pth"
echo "   - 其他必要文件"
echo "3. 将下载的文件放置到: $MODELS_DIR/indextts2/"
echo ""
echo "目录结构应该如下："
echo "models/indextts2/"
echo "  ├── config.json"
echo "  ├── checkpoint.pth"
echo "  └── ..."
echo ""

# 检查IndexTTS2模型是否已存在
if [ -f "$MODELS_DIR/indextts2/config.json" ] && [ -f "$MODELS_DIR/indextts2/checkpoint.pth" ]; then
    echo -e "${GREEN}✓ IndexTTS2模型文件已存在${NC}"
else
    echo -e "${YELLOW}⚠ IndexTTS2模型文件未找到，请手动下载${NC}"
fi

echo ""
echo "======================================"
echo "模型下载检查完成"
echo "======================================"
echo ""
echo "提示："
echo "- FunASR会在首次运行时自动下载"
echo "- IndexTTS2需要手动下载并放置到正确位置"
echo "- 运行 'python main.py' 开始使用系统"
echo ""

