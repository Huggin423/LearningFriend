# IndexTTS2 语音合成模块

IndexTTS2 语音合成系统，使用 ModelScope 官方预训练模型。

## ✨ 特性

- ✅ **零样本语音克隆**：从短参考音频克隆音色
- ✅ **情感控制**：支持7种基本情感和自然语言描述
- ✅ **高质量合成**：使用官方预训练模型，音质优秀
- ✅ **GPU/CPU支持**：自动适配设备
- ✅ **ModelScope集成**：自动从国内镜像下载，速度快

## 🚀 快速开始

### 步骤1：安装依赖

```bash
pip install modelscope
```

### 步骤2：配置模型路径

编辑 `config/config.yaml`:

```yaml
tts:
  model_path: "models/indextts2"  # 模型保存路径
  device: "cuda"                  # 使用GPU
  sample_rate: 22050
  speed: 1.0
  emotion: "neutral"
```

### 步骤3：手动下载模型（可选）

如果首次运行自动下载较慢，可以预先下载：

```bash
python scripts/download_indextts2_modelscope.py
```

### 步骤4：测试使用

```python
from config import load_config
from src.tts import create_tts_module

config = load_config()
tts = create_tts_module(config['tts'])

# 测试合成
audio = tts.synthesize("你好，这是IndexTTS2测试")
print(f"音频长度: {len(audio)/22050:.2f}秒")
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
- 格式：WAV（推荐22.05kHz）
- 质量：清晰无噪音

#### 3. 批量合成

```python
texts = ["你好", "谢谢", "再见"]
audios = tts.synthesize_batch(texts)
```

#### 4. 语速控制

```python
speeds = [0.8, 1.0, 1.2, 1.5]

for speed in speeds:
    audio = tts.synthesize(
        text="语速控制测试",
        speed=speed
    )
```

## ⚙️ 配置参数

```yaml
tts:
  # 模型路径
  model_path: "models/indextts2"
  
  # 设备配置
  device: "cuda"  # 或 "cpu"
  
  # 音频参数
  sample_rate: 22050
  speed: 1.0
  
  # 音色设置
  speaker_id: 0
  pitch: 1.0
  
  # 情感设置
  emotion: "neutral"
```

## 🎯 与Pipeline集成

系统已经自动集成，无需修改代码！

`ConversationPipeline` 会自动加载TTS模块：

```python
from src.pipeline import ConversationPipeline
from config import load_config

config = load_config()
pipeline = ConversationPipeline(config)

# 使用TTS进行完整对话
result = pipeline.process_audio_file("input.wav")
```

## 🔧 故障排除

### 问题1：模型下载慢

**解决**：使用国内 ModelScope 镜像，速度更快

```python
# 或在代码中设置镜像地址
import os
os.environ['MODELSCOPE_ENVIRONMENT'] = 'aliyun'
```

### 问题2：ModelScope未安装

**解决**：
```bash
pip install modelscope
```

### 问题3：CUDA out of memory

**解决**：
```yaml
# 使用CPU
device: "cpu"
```

### 问题4：音质不好

**解决方案**：
1. 提供高质量参考音频（3-10秒，清晰无噪音）
2. 调整情感强度
```python
emotion_strength: 0.7  # 降低强度
```

### 问题5：导入失败

**检查清单**：
- [ ] ModelScope 已安装
- [ ] 模型文件已下载
- [ ] 配置文件路径正确

## 📦 依赖

```bash
pip install modelscope torch torchaudio librosa soundfile
```

## 📚 相关资源

- **ModelScope模型**: https://modelscope.cn/models/IndexTeam/IndexTTS-2
- **官方代码**: https://github.com/index-tts/index-tts
- **论文**: https://arxiv.org/abs/2506.21619

## 📝 架构说明

### ModelScope Pipeline

系统使用 ModelScope 的 pipeline API 加载和使用模型：

1. **自动下载**：首次使用时自动从 ModelScope 下载模型
2. **缓存管理**：模型文件缓存到本地，下次直接使用
3. **设备适配**：自动适配 GPU/CPU 设备

### 模型文件结构

下载后的模型文件结构：
```
models/indextts2/
├── config.yaml              # 配置文件
├── bpe.model                # BPE tokenizer
├── feat1.pt                 # 特征文件1
├── feat2.pt                 # 特征文件2
└── qwen0.6bemo4-merge/      # Qwen情感模型
    └── model.safetensors    # 模型权重
```

## ✅ 检查清单

使用前请确认：

- [ ] ModelScope 已安装 (`pip install modelscope`)
- [ ] 配置文件已正确设置
- [ ] GPU 显存足够（推荐8GB以上）
- [ ] 测试合成成功

## ⚠️ 注意事项

1. **首次使用**：会自动下载模型（约5.9GB，需要时间）
2. **GPU内存**：推荐至少8GB显存
3. **网络**：需要从ModelScope下载模型
4. **国内用户**：ModelScope镜像速度快，推荐使用

## 🔄 版本历史

- **v2.0**: 重构为仅使用 ModelScope 官方模型
- **v1.0**: 支持官方模型和复现模型两种方式

## 📄 许可证

MIT License
