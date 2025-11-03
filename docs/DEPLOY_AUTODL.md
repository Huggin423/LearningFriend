# AutoDL 部署指南

本文档介绍如何在 AutoDL 平台上部署 LearningFriend 项目。

## 🔧 前置准备

### 1. 安装系统依赖

在安装 Python 包之前，需要先安装系统级依赖：

```bash
# 更新包管理器
apt-get update

# 安装必要的系统库
apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    gcc \
    g++ \
    make
```

**注意**：`pyaudio` 需要 `portaudio19-dev`，但如果你不需要实时录音功能，可以跳过安装 pyaudio。

### 2. 安装 Python 依赖

```bash
# 进入项目目录
cd ~/LearningFriend

# 安装核心依赖（跳过 pyaudio，它在 requirements.txt 中已被注释）
pip install -r requirements.txt

# 如果 torchaudio 安装失败，可以单独安装
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装 FunASR

```bash
cd ~/LearningFriend/FunASR
pip install -e .
cd ..
```

或者使用提供的脚本：

```bash
bash scripts/setup_funasr.sh
```

## 📋 完整安装脚本

创建一个安装脚本 `scripts/install_autodl.sh`：

```bash
#!/bin/bash
set -e

echo "======================================"
echo "AutoDL 环境安装脚本"
echo "======================================"

# 1. 安装系统依赖（可选，如果不需要 pyaudio 可以跳过）
echo "[1/5] 安装系统依赖..."
apt-get update
apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    gcc \
    g++ \
    make || echo "⚠ 系统依赖安装失败（如果是容器环境可能是正常的）"

# 2. 安装 Python 核心依赖
echo "[2/5] 安装 Python 核心依赖..."
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

# 3. 安装 FunASR
echo "[3/5] 安装 FunASR..."
cd FunASR
pip install -e .
cd ..

# 4. 验证安装
echo "[4/5] 验证安装..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import torchaudio; print(f'✓ torchaudio {torchaudio.__version__}')" || echo "⚠ torchaudio 未安装"
python -c "import funasr; print('✓ FunASR 安装成功')" || echo "⚠ FunASR 未正确安装"
python -c "import librosa; print(f'✓ librosa {librosa.__version__}')"
python -c "import soundfile; print(f'✓ soundfile {soundfile.__version__}')"

# 5. 测试导入
echo "[5/5] 测试模块导入..."
python -c "from src.asr import FunASRModule; print('✓ ASR 模块导入成功')" || echo "⚠ ASR 模块导入失败"
python -c "from src.llm import LLMInterface; print('✓ LLM 模块导入成功')" || echo "⚠ LLM 模块导入失败"

echo ""
echo "======================================"
echo "安装完成！"
echo "======================================"
echo ""
echo "下一步："
echo "1. 配置 API Key: cp config/config.yaml.example config/config.yaml"
echo "2. 编辑配置文件，填入你的 API Key"
echo "3. 运行测试: python test_pipeline.py"
```

使用方法：

```bash
chmod +x scripts/install_autodl.sh
bash scripts/install_autodl.sh
```

## 🐛 常见问题

### 问题1: pyaudio 构建失败

**错误信息**：
```
fatal error: portaudio.h: No such file or directory
```

**解决方案**：

选项 A（推荐）：不需要实时录音功能时，跳过安装 pyaudio
- `requirements.txt` 中 `pyaudio` 已被注释
- 直接安装其他依赖即可

选项 B：需要实时录音功能时，先安装系统依赖
```bash
apt-get update
apt-get install -y portaudio19-dev libasound2-dev
pip install pyaudio
```

### 问题2: torchaudio 未安装

**错误信息**：
```
ModuleNotFoundError: No module named 'torchaudio'
```

**解决方案**：
```bash
# 方法1：单独安装 torchaudio
pip install torchaudio

# 方法2：如果方法1失败，指定 PyTorch 索引
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 方法3：对于 CUDA 11.8 环境
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题3: logger 未定义错误

**错误信息**：
```
NameError: name 'logger' is not defined
```

**解决方案**：
此问题已在最新版本中修复。如果仍然出现，请确保使用最新代码：

```bash
git pull origin master
```

或者手动修复 `src/asr/funasr_module.py`，确保 `logger = logging.getLogger(__name__)` 在 `try-except` 块之前定义。

### 问题4: FunASR SyntaxWarning

**警告信息**：
```
SyntaxWarning: invalid escape sequence '\['
```

**解决方案**：
这些警告来自 FunASR 源码，不影响功能。如果想消除警告，可以：

```bash
# 方法1：运行时忽略警告
python -W ignore::SyntaxWarning test_pipeline.py

# 方法2：在代码中设置
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
```

### 问题5: 模型下载慢

**解决方案**：
```bash
# 设置 ModelScope 镜像（国内用户）
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache

# 或者使用代理（如果有）
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
```

## ✅ 验证安装

运行测试脚本：

```bash
python test_pipeline.py
```

应该看到：
- ✓ ASR 模块初始化成功
- ✓ LLM 模块初始化成功（如果配置了 API Key）
- ✓ TTS 模块初始化成功
- ✓ 测试通过

## 🚀 开始使用

安装完成后：

1. **配置 API Key**
   ```bash
   cp config/config.yaml.example config/config.yaml
   # 编辑 config/config.yaml，填入 API Key
   ```

2. **运行测试**
   ```bash
   python test_pipeline.py
   ```

3. **开始对话**
   ```bash
   python main.py --mode interactive
   ```

## 📝 注意事项

1. **GPU 环境**：AutoDL 通常提供 GPU 环境，确保在配置中使用 `device: "cuda"`
2. **存储空间**：FunASR 模型约 1-2GB，确保有足够空间
3. **网络连接**：首次运行需要下载模型，确保网络畅通
4. **pyaudio**：服务器环境通常不需要实时录音，可以跳过安装

## 🔗 相关链接

- [AutoDL 官方文档](https://www.autodl.com/docs/)
- [FunASR 文档](https://github.com/alibaba-damo-academy/FunASR)
- [项目主 README](../README.md)

