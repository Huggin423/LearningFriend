# IndexTTS2 语音合成模块

IndexTTS2 语音合成系统，支持官方模型和复现模型两种实现方式。

## ✨ 特性

- ✅ **零样本语音克隆**：从短参考音频克隆音色
- ✅ **情感控制**：支持7种基本情感和自然语言描述
- ✅ **精确时长控制**：指定语义token数量实现精确控制
- ✅ **自然韵律**：自由生成模式保留韵律特征
- ✅ **GPU/CPU支持**：自动适配设备
- ✅ **官方模型支持**：可直接使用HuggingFace预训练模型

## 📋 两种实现方式

### 方式1：官方模型（推荐 ⭐）

**优点**：
- ✅ 开箱即用，无需训练
- ✅ 高质量预训练权重
- ✅ 完整功能支持
- ✅ 官方维护更新

**缺点**：
- ⚠️ 需要下载约5.9GB模型
- ⚠️ 需要克隆官方代码仓库

### 方式2：复现模型

**优点**：
- ✅ 完整理解架构
- ✅ 可自定义训练
- ✅ 不依赖外部代码

**缺点**：
- ⚠️ 需要预训练权重（当前未发布）
- ⚠️ 需要自己训练

## 🚀 快速开始

### 使用官方模型（推荐）

#### 步骤1：更新配置

编辑 `config/config.yaml`:

```yaml
tts:
  use_official: true  # 使用官方模型
  
  # 官方模型配置
  official_repo: "index-tts"      # 官方代码路径
  model_path: "checkpoints"       # 模型保存路径
  device: "cuda"                  # 使用GPU
```

#### 步骤2：自动安装

运行一键安装脚本：

```bash
bash src/tts/setup_indextts2.sh
```

这个脚本会自动：
1. ✅ 克隆官方代码仓库
2. ✅ 安装所有依赖
3. ✅ 下载模型文件（约5.9GB）
4. ✅ 验证安装

#### 步骤3：手动安装（可选）

如果自动脚本失败，可以手动安装：

```bash
# 1. 克隆官方仓库
git clone https://github.com/index-tts/index-tts.git index-tts

# 2. 安装依赖
cd index-tts
pip install -r requirements.txt
cd ..

# 3. 下载模型
pip install huggingface-hub
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints
```

#### 步骤4：测试

```python
from config import load_config
from src.tts import create_tts_module

config = load_config()
tts = create_tts_module(config['tts'])

# 测试合成
audio = tts.synthesize("你好，这是IndexTTS2测试")
print(f"音频长度: {len(audio)/22050:.2f}秒")
```

### 使用复现模型

```yaml
tts:
  use_official: false
  model_path: "models/indextts2"
  device: "cuda"
  # ... 其他复现模型配置
```

## 💻 使用方法

### 基本使用

```python
from config import load_config
from src.tts import create_tts_module

# 加载配置
config = load_config()
tts_config = config['tts']

# 初始化模块
tts = create_tts_module(tts_config)

# 合成语音
audio = tts.synthesize("你好，这是一个测试")

# 保存到文件
tts.synthesize_to_file("你好", "output.wav")
```

### 高级功能

#### 1. 情感控制

```python
# 使用预定义情感
audio = tts.synthesize("你好", emotion="happiness")

# 使用自然语言描述
audio = tts.synthesize("你好", emotion="愉悦开心")

# 设置情感强度
audio = tts.synthesize(
    text="今天心情很好！",
    emotion="happiness",
    emotion_strength=0.8
)
```

支持的情感：
```python
emotions = [
    'neutral',      # 中性
    'happiness',    # 开心
    'sadness',      # 悲伤
    'anger',        # 愤怒
    'fear',         # 恐惧
    'disgust',      # 厌恶
    'surprise'      # 惊讶
]
```

#### 2. 语音克隆

```python
import soundfile as sf

# 加载参考音频
ref_audio, sr = sf.read("reference.wav")

# 零样本语音克隆
cloned_audio = tts.clone_voice(
    reference_audio=ref_audio,
    text="新的语音内容",
    emotion="neutral"
)
```

**参考音频要求**：
- 时长：3-10秒
- 格式：WAV（推荐16kHz或22.05kHz）
- 质量：清晰无噪音

#### 3. 时长控制

```python
# 精确时长控制（指定秒数）
audio = tts.synthesize("你好", target_duration=2.0)

# 自由生成模式
audio = tts.synthesize("你好", target_duration=None)
```

**注意**：官方模型当前版本时长控制功能尚未启用。

#### 4. 批量合成

```python
texts = ["你好", "谢谢", "再见"]
audios = tts.synthesize_batch(texts)
```

#### 5. 语速控制

```python
speeds = [0.8, 1.0, 1.2, 1.5]

for speed in speeds:
    audio = tts.synthesize(
        text="语速控制测试",
        speed=speed
    )
```

## ⚙️ 配置参数

### 官方模型配置

```yaml
tts:
  use_official: true
  
  # 路径配置
  official_repo: "index-tts"          # 官方代码仓库路径
  model_path: "checkpoints"           # 模型文件路径
  
  # 设备配置
  device: "cuda"  # 或 "cpu"
  
  # 音频参数
  sample_rate: 22050
  speed: 1.0
  
  # 情感设置
  emotion: "neutral"
  emotion_strength: 1.0
```

### 复现模型配置

```yaml
tts:
  use_official: false
  
  # 模型路径
  model_path: "models/indextts2"
  device: "cuda"
  speaker_id: 0
  speed: 1.0
  pitch: 1.0
  sample_rate: 22050
  
  # 模型检查点
  t2s_checkpoint: "models/indextts2/t2s_model.pth"
  s2m_checkpoint: "models/indextts2/s2m_model.pth"
  vocoder_checkpoint: "models/indextts2/vocoder.pth"
  t2e_checkpoint: "models/indextts2/t2e_model.pth"
  
  # 模型参数
  t2s_d_model: 512
  t2s_n_heads: 8
  t2s_n_layers: 12
  max_seq_length: 2048
  
  # 功能开关
  enable_emotion_control: true
```

## 📦 模块结构

### 核心模块

1. **Text-to-Semantic (T2S)**
   - 位置: `models/text_to_semantic.py`
   - 功能: 自回归生成语义token
   - 创新: 时长控制 + 情感解耦

2. **Semantic-to-Mel (S2M)**
   - 位置: `models/semantic_to_mel.py`
   - 功能: 流匹配生成Mel频谱
   - 创新: GPT隐层增强

3. **Vocoder (BigVGANv2)**
   - 位置: `models/vocoder.py`
   - 功能: Mel频谱转音频波形
   - 特点: 高保真重建

4. **Text-to-Emotion (T2E)**
   - 位置: `models/text_to_emotion.py`
   - 功能: 自然语言情感映射
   - 创新: LoRA知识蒸馏

### 工具模块

- `utils/audio_utils.py`: 音频处理工具
- `utils/text_utils.py`: 文本分词和规范化

## 🔄 切换模型

### 使用官方模型（默认）

```yaml
tts:
  use_official: true
```

### 使用复现模型

```yaml
tts:
  use_official: false
  model_path: "models/indextts2"
```

系统会自动根据配置选择对应的实现方式。如果官方模型加载失败，会自动回退到复现模型。

## 🎯 与Pipeline集成

系统已经自动集成，无需修改代码！

`ConversationPipeline` 会自动：
1. 读取配置中的 `use_official` 选项
2. 选择使用官方模型或复现模型
3. 如果官方模型不可用，自动回退到复现模型

```python
from src.pipeline import ConversationPipeline
from config import load_config

config = load_config()
pipeline = ConversationPipeline(config)

# 使用官方模型进行完整对话
result = pipeline.process_audio_file("input.wav")
```

## 📊 功能对比

| 功能 | 官方模型 | 复现模型 |
|------|---------|---------|
| 语音合成 | ✅ | ⚠️ 需权重 |
| 零样本克隆 | ✅ | ✅ |
| 情感控制 | ✅ | ✅ |
| 时长控制 | ⚠️ 待更新 | ✅ |
| 批量处理 | ✅ | ✅ |
| 自定义训练 | ❌ | ✅ |

## 🔧 故障排除

### 问题1：官方模型下载慢

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints
```

### 问题2：导入失败

```python
# 确保路径正确
import sys
sys.path.insert(0, 'index-tts')

# 或设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/index-tts"
```

### 问题3：CUDA out of memory

```yaml
# 1. 使用CPU
device: "cpu"

# 2. 启用FP16（如果支持）
use_fp16: true

# 3. 减少batch size
batch_size: 1
```

### 问题4：音质不好

**解决**:
```python
# 1. 提供高质量参考音频（3-10秒，清晰无噪音）
# 2. 调整情感强度
emotion_strength: 0.7  # 降低强度

# 3. 使用更好的采样参数
temperature: 0.8
top_k: 50
```

### 问题5：自动回退

如果官方模型加载失败，系统会自动回退到复现模型（返回静音占位）。检查：
1. 官方代码是否正确克隆
2. 模型文件是否完整下载
3. 依赖是否正确安装

## 📚 相关资源

- **官方模型**: https://huggingface.co/IndexTeam/IndexTTS-2
- **官方代码**: https://github.com/index-tts/index-tts
- **在线Demo**: https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo
- **论文**: https://arxiv.org/abs/2506.21619

## ✅ 检查清单

### 官方模型

- [ ] 已设置 `use_official: true`
- [ ] 已运行安装脚本或手动安装
- [ ] 模型文件已下载（checkpoints目录）
- [ ] 官方代码已克隆（index-tts目录）
- [ ] 测试合成成功

### 复现模型

- [ ] 已设置 `use_official: false`
- [ ] 模型权重文件已准备（如果需要）
- [ ] 配置文件已正确设置
- [ ] 测试合成成功

## ⚠️ 注意事项

### 官方模型

1. **首次使用**：会自动下载模型（约5.9GB，需要时间）
2. **GPU内存**：推荐至少8GB显存
3. **依赖**：需要安装官方代码和依赖
4. **网络**：需要从HuggingFace下载模型

### 复现模型

1. **预训练权重**：需要等官方发布或自己训练
2. **当前状态**：架构完整，但无权重
3. **学习价值**：完整理解论文实现

## 📖 依赖

```bash
pip install torch torchaudio librosa soundfile
pip install huggingface-hub  # 官方模型需要
```

## 📄 许可证

MIT License
