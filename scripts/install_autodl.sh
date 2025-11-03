#!/bin/bash
set -e

echo "======================================"
echo "AutoDL 环境安装脚本"
echo "======================================"

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 1. 安装系统依赖（可选，如果不需要 pyaudio 可以跳过）
echo -e "${GREEN}[1/5] 安装系统依赖...${NC}"
if command -v apt-get &> /dev/null; then
    apt-get update || echo -e "${YELLOW}⚠ 无法更新 apt（可能是容器环境）${NC}"
    apt-get install -y \
        portaudio19-dev \
        libasound2-dev \
        gcc \
        g++ \
        make 2>/dev/null || echo -e "${YELLOW}⚠ 系统依赖安装失败（如果是容器环境可能是正常的）${NC}"
else
    echo -e "${YELLOW}⚠ 未找到 apt-get，跳过系统依赖安装${NC}"
fi

# 2. 安装 Python 核心依赖
echo -e "${GREEN}[2/5] 安装 Python 核心依赖...${NC}"
pip install --upgrade pip
pip install \
    pyyaml>=6.0 \
    numpy>=1.24.0 \
    torch>=2.0.0 \
    torchaudio>=2.0.0 \
    modelscope>=1.9.0 \
    openai>=1.0.0 \
    requests>=2.31.0 \
    scipy>=1.10.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    huggingface-hub>=0.19.0 \
    webrtcvad>=2.0.10 \
    tqdm>=4.65.0 \
    loguru>=0.7.0

# 如果 torchaudio 安装失败，尝试单独安装
python -c "import torchaudio" 2>/dev/null || {
    echo -e "${YELLOW}⚠ torchaudio 未正确安装，尝试单独安装...${NC}"
    pip install torchaudio || {
        echo -e "${YELLOW}⚠ 尝试从 PyTorch 官方源安装...${NC}"
        pip install torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
        pip install torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
}

# 3. 安装 FunASR
echo -e "${GREEN}[3/5] 安装 FunASR...${NC}"
if [ -d "FunASR" ]; then
    cd FunASR
    pip install -e .
    cd ..
    echo -e "${GREEN}✓ FunASR 安装完成${NC}"
else
    echo -e "${RED}✗ FunASR 目录不存在，请先克隆 FunASR 仓库${NC}"
    echo "  git clone https://github.com/alibaba-damo-academy/FunASR.git"
    exit 1
fi

# 4. 验证安装
echo -e "${GREEN}[4/5] 验证安装...${NC}"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo -e "${RED}✗ PyTorch 未安装${NC}"
python -c "import torchaudio; print(f'✓ torchaudio {torchaudio.__version__}')" || echo -e "${YELLOW}⚠ torchaudio 未安装${NC}"
python -c "import funasr; print('✓ FunASR 安装成功')" || echo -e "${YELLOW}⚠ FunASR 未正确安装${NC}"
python -c "import librosa; print(f'✓ librosa {librosa.__version__}')" || echo -e "${RED}✗ librosa 未安装${NC}"
python -c "import soundfile; print(f'✓ soundfile {soundfile.__version__}')" || echo -e "${RED}✗ soundfile 未安装${NC}"

# 5. 测试导入
echo -e "${GREEN}[5/5] 测试模块导入...${NC}"
python -c "from src.asr import FunASRModule; print('✓ ASR 模块导入成功')" || echo -e "${YELLOW}⚠ ASR 模块导入失败${NC}"
python -c "from src.llm import LLMInterface; print('✓ LLM 模块导入成功')" || echo -e "${YELLOW}⚠ LLM 模块导入失败${NC}"
python -c "from src.tts import IndexTTSModule; print('✓ TTS 模块导入成功')" || echo -e "${YELLOW}⚠ TTS 模块导入失败${NC}"

echo ""
echo "======================================"
echo -e "${GREEN}安装完成！${NC}"
echo "======================================"
echo ""
echo "下一步："
echo "1. 配置 API Key: cp config/config.yaml.example config/config.yaml"
echo "2. 编辑配置文件，填入你的 API Key"
echo "3. 运行测试: python test_pipeline.py"
echo ""

