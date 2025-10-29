#!/bin/bash
# FunASR环境设置脚本

echo "======================================"
echo "FunASR环境设置"
echo "======================================"

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUNASR_DIR="$PROJECT_ROOT/FunASR"

echo -e "${YELLOW}项目根目录: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}FunASR目录: $FUNASR_DIR${NC}"
echo ""

# 检查FunASR是否存在
if [ ! -d "$FUNASR_DIR" ]; then
    echo -e "${RED}✗ FunASR目录不存在: $FUNASR_DIR${NC}"
    echo "请先克隆FunASR仓库："
    echo "  git clone https://github.com/alibaba-damo-academy/FunASR.git"
    exit 1
fi

echo -e "${GREEN}✓ 找到FunASR目录${NC}"
echo ""

# 安装FunASR
echo -e "${GREEN}[1/3] 安装FunASR...${NC}"
cd "$FUNASR_DIR"

if command -v pip &> /dev/null; then
    echo "使用pip安装FunASR..."
    pip install -e .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ FunASR安装成功${NC}"
    else
        echo -e "${RED}✗ FunASR安装失败${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ 未找到pip命令${NC}"
    exit 1
fi

echo ""

# 安装ModelScope
echo -e "${GREEN}[2/3] 安装ModelScope...${NC}"
pip install modelscope
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ModelScope安装成功${NC}"
else
    echo -e "${YELLOW}⚠ ModelScope安装可能出现问题${NC}"
fi

echo ""

# 测试导入
echo -e "${GREEN}[3/3] 测试FunASR导入...${NC}"
python -c "import funasr; print('FunASR版本:', funasr.__version__)"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ FunASR导入测试成功${NC}"
else
    echo -e "${RED}✗ FunASR导入测试失败${NC}"
    exit 1
fi

echo ""
echo "======================================"
echo "FunASR环境设置完成"
echo "======================================"
echo ""
echo "下一步："
echo "1. 返回项目根目录: cd $PROJECT_ROOT"
echo "2. 运行系统: python main.py"
echo ""

