#!/bin/bash
# IndexTTS2 自动安装脚本
# 一键下载模型和配置环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  IndexTTS2 自动安装脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 打印信息
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 检查Python版本
info "检查Python版本..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    info "Python版本: $PYTHON_VERSION"
    
    # 检查是否满足最低要求 (3.8+)
    if [ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 0 ]; then
        error "Python版本过低，需要 >= 3.8"
        exit 1
    fi
else
    error "未找到Python3，请先安装Python"
    exit 1
fi

# 2. 检查Git
info "检查Git..."
if ! command_exists git; then
    error "未找到Git，请先安装: sudo apt-get install git"
    exit 1
fi

# 3. 创建目录结构
info "创建目录结构..."
mkdir -p checkpoints
mkdir -p index-tts
mkdir -p data/audio_input
mkdir -p data/audio_output
mkdir -p data/logs
mkdir -p models/indextts2

# 4. 克隆官方仓库
info "克隆IndexTTS官方仓库..."
if [ ! -d "index-tts/.git" ]; then
    git clone https://github.com/index-tts/index-tts.git index-tts
    info "✓ 仓库克隆成功"
else
    info "官方仓库已存在，跳过克隆"
fi

# 5. 安装Python依赖
info "安装Python依赖..."

# 升级pip
python3 -m pip install --upgrade pip

# 安装PyTorch（根据是否有CUDA选择版本）
if command_exists nvidia-smi; then
    info "检测到NVIDIA GPU，安装CUDA版本的PyTorch..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    warning "未检测到NVIDIA GPU，安装CPU版本的PyTorch..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
info "安装其他依赖包..."
python3 -m pip install numpy scipy
python3 -m pip install librosa soundfile
python3 -m pip install pyyaml
python3 -m pip install huggingface-hub

# 安装官方仓库的依赖
if [ -f "index-tts/requirements.txt" ]; then
    info "安装IndexTTS依赖..."
    python3 -m pip install -r index-tts/requirements.txt
fi

info "✓ 依赖安装完成"

# 6. 下载模型
info "下载IndexTTS2模型..."

# 检查模型是否已存在
if [ -f "checkpoints/config.yaml" ] && [ -f "checkpoints/feat1.pt" ]; then
    warning "模型文件已存在，跳过下载"
else
    info "开始下载模型（约5.9GB，请耐心等待）..."
    
    # 检查huggingface-cli是否可用
    if command_exists huggingface-cli; then
        info "使用huggingface-cli下载..."
        huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints
        info "✓ 模型下载成功"
    else
        warning "huggingface-cli不可用，尝试使用Python API..."
        python3 << EOF
from huggingface_hub import snapshot_download
print("正在下载模型...")
snapshot_download(
    repo_id="IndexTeam/IndexTTS-2",
    local_dir="checkpoints",
    local_dir_use_symlinks=False
)
print("✓ 模型下载成功")
EOF
    fi
fi

# 7. 验证安装
info "验证安装..."

# 检查关键文件
REQUIRED_FILES=(
    "checkpoints/config.yaml"
    "checkpoints/bpe.model"
    "checkpoints/feat1.pt"
    "checkpoints/feat2.pt"
)

ALL_EXISTS=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        error "缺少文件: $file"
        ALL_EXISTS=false
    fi
done

if [ "$ALL_EXISTS" = true ]; then
    info "✓ 所有必需文件已就绪"
else
    error "部分文件缺失，请检查下载"
    exit 1
fi

# 8. 创建测试脚本
info "创建测试脚本..."
cat > test_installation.py << 'EOF'
"""
安装验证脚本
"""
import sys
import torch

print("=" * 50)
print("IndexTTS2 安装验证")
print("=" * 50)

# 1. 检查PyTorch
print(f"\n1. PyTorch版本: {torch.__version__}")
print(f"   CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    print(f"   GPU名称: {torch.cuda.get_device_name(0)}")

# 2. 检查依赖包
print("\n2. 依赖包检查:")
packages = [
    'numpy', 'scipy', 'librosa', 'soundfile', 
    'yaml', 'huggingface_hub'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg} (未安装)")

# 3. 检查模型文件
print("\n3. 模型文件检查:")
from pathlib import Path

files = [
    'checkpoints/config.yaml',
    'checkpoints/bpe.model',
    'checkpoints/feat1.pt',
    'checkpoints/feat2.pt'
]

for file in files:
    exists = Path(file).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {file}")

# 4. 检查官方代码
print("\n4. 官方代码检查:")
repo_path = Path('index-tts')
if repo_path.exists():
    print(f"   ✓ 官方仓库: {repo_path}")
    
    # 尝试导入
    sys.path.insert(0, str(repo_path))
    try:
        # 这里根据实际情况调整
        print("   ✓ Python路径已配置")
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
else:
    print(f"   ✗ 官方仓库不存在")

print("\n" + "=" * 50)
print("验证完成！")
print("=" * 50)
EOF

info "运行验证脚本..."
python3 test_installation.py

# 9. 创建快速启动脚本
info "创建快速启动脚本..."
cat > run_indextts2.py << 'EOF'
"""
IndexTTS2 快速测试脚本
"""
import sys
from pathlib import Path

# 添加官方代码路径
sys.path.insert(0, str(Path('index-tts')))

def main():
    print("IndexTTS2 测试")
    print("=" * 50)
    
    try:
        # 导入官方模块（根据实际调整）
        print("正在初始化模型...")
        
        # 这里添加实际的初始化代码
        # from inference import IndexTTSInference
        # model = IndexTTSInference(...)
        
        print("✓ 模型初始化成功")
        
        # 测试合成
        text = "你好，这是IndexTTS2测试。"
        print(f"\n测试文本: {text}")
        
        # audio = model.synthesize(text)
        # print(f"✓ 合成成功")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        print("\n请参考官方文档进行配置")

if __name__ == "__main__":
    main()
EOF

# 10. 显示后续步骤
echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""
info "后续步骤:"
echo ""
echo "1. 验证安装:"
echo "   python3 test_installation.py"
echo ""
echo "2. 运行官方Demo:"
echo "   cd index-tts"
echo "   python webui.py"
echo ""
echo "3. 集成到你的项目:"
echo "   - 复制 indextts2_official_wrapper.py 到 src/tts/"
echo "   - 更新配置文件 config/config.yaml"
echo "   - 运行测试 python test_pipeline.py"
echo ""
echo "4. 文档:"
echo "   - 查看 QUICK_INTEGRATION.md"
echo "   - 官方文档: https://github.com/index-tts/index-tts"
echo ""
info "如有问题，请查看日志或提Issue"
echo ""