#!/bin/bash
# ===================================================================
# 智能学伴系统 - 一键环境配置脚本
# LearningFriend - Complete Setup Script
# ===================================================================
# 功能：自动完成所有环境配置，包括：
#   1. 检查Python环境
#   2. 安装系统依赖
#   3. 安装Python核心依赖
#   4. 安装FunASR
#   5. 安装IndexTTS2官方代码
#   6. 下载模型文件
#   7. 配置文件初始化
#   8. 验证安装
# ===================================================================

set -e  # 遇到错误立即退出

# ============= 颜色设置 =============
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============= 获取项目根目录 =============
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo -e "${CYAN}智能学伴系统 - 一键环境配置脚本${NC}"
echo "=========================================="
echo -e "${BLUE}项目根目录: $PROJECT_ROOT${NC}"
echo ""

# ============= 步骤计数器 =============
STEP=0
TOTAL_STEPS=8

# ============= 辅助函数 =============
log_info() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${GREEN}[$STEP/$TOTAL_STEPS] $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠  $1${NC}"
}

log_error() {
    echo -e "${RED}✗  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓  $1${NC}"
}

prompt_user() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# ============= 1. 检查Python环境 =============
log_info "检查Python环境"
if ! command -v python3 &> /dev/null; then
    log_error "Python3 未安装"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "  版本: $PYTHON_VERSION"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    log_success "Python版本符合要求 (>= 3.8)"
else
    log_error "Python版本过低，需要 >= 3.8"
    exit 1
fi

# 检查pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    log_error "pip 未安装"
    exit 1
fi

PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi
log_success "pip可用"

# ============= 2. 安装系统依赖 =============
log_info "安装系统依赖"
if command -v apt-get &> /dev/null; then
    echo "检测到apt-get，安装系统依赖..."
    apt-get update 2>/dev/null || log_warn "无法更新apt（可能是容器环境）"
    
    apt-get install -y \
        portaudio19-dev \
        libasound2-dev \
        gcc \
        g++ \
        make \
        git 2>/dev/null || log_warn "系统依赖安装失败（如果是容器环境可能是正常的）"
    
    log_success "系统依赖安装完成"
else
    log_warn "未检测到apt-get，跳过系统依赖安装（Linux/Debian系统需要）"
fi

# ============= 3. 安装Python核心依赖 =============
log_info "安装Python核心依赖"
echo "升级pip..."
python3 -m pip install --upgrade pip -q

echo "安装核心依赖包..."
python3 -m pip install \
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
    loguru>=0.7.0 \
    transformers==4.52.1 \
    omegaconf>=2.3.0 \
    safetensors>=0.3.0 \
    sentencepiece>=0.2.1 \
    einops>=0.8.1 \
    accelerate>=1.0.0 \
    addict>=2.4.0

# 验证torchaudio
if python3 -c "import torchaudio" 2>/dev/null; then
    log_success "PyTorch和torchaudio安装成功"
else
    log_warn "torchaudio未正确安装，尝试单独安装..."
    python3 -m pip install torchaudio || {
        log_warn "从PyTorch官方源安装torchaudio..."
        python3 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
        python3 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
fi

# ============= 4. 克隆FunASR仓库并安装 =============
log_info "安装FunASR"
FUNASR_DIR="$PROJECT_ROOT/FunASR"

if [ ! -d "$FUNASR_DIR" ]; then
    echo "FunASR目录不存在，需要克隆仓库..."
    
    if prompt_user "是否克隆FunASR仓库？(约500MB)"; then
        echo "克隆FunASR仓库..."
        git clone https://github.com/alibaba-damo-academy/FunASR.git "$FUNASR_DIR" || {
            log_error "克隆FunASR仓库失败"
            exit 1
        }
        log_success "FunASR仓库克隆成功"
    else
        log_warn "跳过FunASR克隆，稍后请手动克隆"
    fi
fi

if [ -d "$FUNASR_DIR" ]; then
    echo "安装FunASR..."
    cd "$FUNASR_DIR"
    python3 -m pip install -e . -q
    cd "$PROJECT_ROOT"
    log_success "FunASR安装完成"
else
    log_warn "FunASR目录不存在，跳过安装"
fi

# ============= 5. 克隆IndexTTS2官方代码 =============
log_info "安装IndexTTS2官方代码"
INDEXTTS_DIR="$PROJECT_ROOT/index-tts"

if [ ! -d "$INDEXTTS_DIR" ]; then
    echo "IndexTTS2目录不存在，需要克隆仓库..."
    
    if prompt_user "是否克隆IndexTTS2官方仓库？"; then
        echo "克隆IndexTTS2官方仓库..."
        git clone https://github.com/index-tts/index-tts.git "$INDEXTTS_DIR" || {
            log_error "克隆IndexTTS2仓库失败"
            exit 1
        }
        log_success "IndexTTS2仓库克隆成功"
        
        # 安装IndexTTS2依赖
        if [ -f "$INDEXTTS_DIR/requirements.txt" ]; then
            echo "安装IndexTTS2依赖..."
            cd "$INDEXTTS_DIR"
            python3 -m pip install -r requirements.txt -q
            cd "$PROJECT_ROOT"
        fi
    else
        log_warn "跳过IndexTTS2克隆"
    fi
else
    log_success "IndexTTS2目录已存在"
fi

cd "$PROJECT_ROOT"

# ============= 6. 下载模型文件 =============
log_info "配置模型文件"

# 创建模型目录
mkdir -p models/funasr
mkdir -p models/indextts2
mkdir -p data/audio_input
mkdir -p data/audio_output
mkdir -p data/logs

echo "FunASR模型会在首次运行时自动下载"
echo "IndexTTS2模型需要手动下载..."

# 检查IndexTTS2模型
INDEXTTS_CHECKPOINTS="$INDEXTTS_DIR/checkpoints"
MODELS_DIR="$PROJECT_ROOT/models/indextts2"

if [ -d "$INDEXTTS_CHECKPOINTS" ] && [ -f "$INDEXTTS_CHECKPOINTS/config.yaml" ]; then
    log_success "检测到IndexTTS2官方模型文件"
    
    if [ ! -f "$MODELS_DIR/config.yaml" ]; then
        if prompt_user "是否将IndexTTS2模型复制到models/indextts2？"; then
            echo "复制模型文件..."
            cp -r "$INDEXTTS_CHECKPOINTS"/* "$MODELS_DIR/" 2>/dev/null || true
            log_success "模型文件复制完成"
        fi
    fi
else
    log_warn "未找到IndexTTS2模型文件"
    
    if prompt_user "是否使用Python脚本下载IndexTTS2模型？(约5.9GB)"; then
        echo "运行下载脚本..."
        python3 scripts/download_indextts2_manual.py || log_warn "模型下载失败，稍后可重试"
    fi
fi

# ============= 7. 配置文件初始化 =============
log_info "初始化配置文件"
CONFIG_FILE="config/config.yaml"
CONFIG_EXAMPLE="config/config.yaml.example"

if [ ! -f "$CONFIG_FILE" ] && [ -f "$CONFIG_EXAMPLE" ]; then
    echo "从示例文件创建配置文件..."
    cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
    log_success "配置文件已创建: $CONFIG_FILE"
    log_warn "请编辑配置文件，填入你的API Key"
else
    if [ -f "$CONFIG_FILE" ]; then
        log_success "配置文件已存在: $CONFIG_FILE"
    else
        log_warn "配置文件不存在，请手动创建"
    fi
fi

# ============= 8. 验证安装 =============
log_info "验证安装"
VALIDATION_PASSED=true

echo "测试模块导入..."
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || { log_error "PyTorch导入失败"; VALIDATION_PASSED=false; }
python3 -c "import torchaudio; print(f'  ✓ torchaudio {torchaudio.__version__}')" || { log_warn "torchaudio导入失败（可能不影响使用）"; }
python3 -c "import librosa; print(f'  ✓ librosa {librosa.__version__}')" || { log_error "librosa导入失败"; VALIDATION_PASSED=false; }
python3 -c "import soundfile; print(f'  ✓ soundfile {soundfile.__version__}')" || { log_error "soundfile导入失败"; VALIDATION_PASSED=false; }

if [ -d "$FUNASR_DIR" ]; then
    python3 -c "import funasr; print('  ✓ FunASR安装成功')" || { log_warn "FunASR导入失败"; }
else
    log_warn "FunASR目录不存在，跳过验证"
fi

python3 -c "from huggingface_hub import snapshot_download; print('  ✓ huggingface-hub可用')" || { log_warn "huggingface-hub不可用"; }
python3 -c "from modelscope.hub.snapshot_download import snapshot_download; print('  ✓ modelscope可用')" || { log_warn "modelscope不可用"; }

# ============= 完成提示 =============
echo ""
echo "=========================================="
if $VALIDATION_PASSED; then
    echo -e "${GREEN}环境配置完成！${NC}"
else
    echo -e "${YELLOW}环境配置基本完成，但部分验证失败${NC}"
fi
echo "=========================================="
echo ""

echo -e "${CYAN}下一步操作：${NC}"
echo "1. 编辑配置文件，填入API Key:"
echo "   ${BLUE}vim config/config.yaml${NC}"
echo "   ${BLUE}或${NC}"
echo "   ${BLUE}nano config/config.yaml${NC}"
echo ""
echo "2. 运行端到端测试:"
echo "   ${BLUE}python3 test_pipeline.py${NC}"
echo ""
echo "3. 开始使用系统:"
echo "   ${BLUE}python3 main.py --mode interactive${NC}"
echo ""
echo -e "${YELLOW}重要提示：${NC}"
echo "• FunASR模型会在首次运行时自动从ModelScope下载"
echo "• 确保配置文件中的API Key已正确填写"
echo "• 如果IndexTTS2模型未下载，会影响TTS功能"
echo "• 查看 README.md 了解详细使用说明"
echo ""

# 如果验证失败，返回非0退出码
exit $([ "$VALIDATION_PASSED" = true ] && echo 0 || echo 1)

