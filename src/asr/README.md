# FunASR 语音识别模块

基于阿里达摩院FunASR的语音识别模块，支持中文和多种语言的高精度语音转文字。

## 特性

- ✅ **高精度识别**：使用Paraformer模型，准确率业界领先
- ✅ **零配置使用**：自动下载模型，开箱即用
- ✅ **多种模型**：支持离线、实时、多语言等模型
- ✅ **长音频支持**：配合VAD可处理任意长度音频
- ✅ **标点恢复**：自动添加标点符号
- ✅ **热词优化**：支持自定义热词提高准确率
- ✅ **GPU/CPU**：自动适配设备

## 快速开始

### 基本使用

```python
from config import load_config
from src.asr import FunASRModule

# 加载配置
config = load_config()
asr_config = config['asr']

# 初始化模块
asr = FunASRModule(asr_config)

# 识别文件
text = asr.transcribe_file("audio.wav")

# 识别数组
import numpy as np
audio_data = np.array([...])  # 你的音频数据
text = asr.transcribe_array(audio_data, sample_rate=16000)
```

### 高级功能

#### 1. 启用VAD和标点恢复

在 `config/config.yaml` 中配置：

```yaml
asr:
  model_name: "paraformer-zh"
  vad_model: "fsmn-vad"          # 启用语音活动检测
  vad_kwargs:
    max_single_segment_time: 60000  # 最大段长60秒
  punc_model: "ct-punc"           # 启用标点恢复
```

#### 2. 设置热词

```python
# 方法1：配置文件
# config.yaml:
#   hotword: "人工智能 机器学习"

# 方法2：代码设置
asr.set_hotword("人工智能 机器学习")
text = asr.transcribe_file("audio.wav")
```

#### 3. 批量识别

```python
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
texts = asr.transcribe_batch(audio_files)
```

#### 4. 使用不同模型

```python
# 实时识别模型
config['asr']['model_name'] = "paraformer-zh-streaming"

# SenseVoice模型（多任务能力）
config['asr']['model_name'] = "iic/SenseVoiceSmall"

# 英文模型
config['asr']['model_name'] = "paraformer-en"
```

## 模型选项

### 推荐的模型

| 模型名称 | 类型 | 描述 | 推荐场景 |
|---------|------|------|---------|
| `paraformer-zh` | 离线 | 中文识别，支持长音频 | ⭐ **推荐**日常使用 |
| `paraformer-zh-streaming` | 实时 | 实时识别，低延迟 | 实时对话 |
| `iic/SenseVoiceSmall` | 离线 | 多任务（ASR+LID+SER+AED） | 复杂场景 |
| `paraformer-en` | 离线 | 英文识别 | 英文识别 |

### 其他模型

- **paraformer**: 基础模型（<=20s）
- **fsmn-vad**: 语音活动检测
- **ct-punc**: 标点恢复
- **cam++**: 说话人确认/分割

完整模型列表：[FunASR Model Zoo](https://github.com/alibaba-damo-academy/FunASR/blob/main/model_zoo/readme_zh.md)

## 配置说明

### 完整配置示例

```yaml
asr:
  model_name: "paraformer-zh"
  model_revision: "v2.0.4"
  device: "cuda"              # 或 "cpu"
  batch_size: 1
  sample_rate: 16000
  hotword: "机器学习 深度学习"
  use_itn: true
  hub: "ms"                   # 或 "hf"
  
  # VAD配置（可选）
  vad_model: "fsmn-vad"
  vad_kwargs:
    max_single_segment_time: 60000
  
  # 标点恢复（可选）
  punc_model: "ct-punc"
  punc_kwargs: {}
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | "paraformer-zh" | 模型名称 |
| `model_revision` | str | "v2.0.4" | 模型版本 |
| `device` | str | "cuda" | 设备，自动检测 |
| `batch_size` | int | 1 | 批处理大小 |
| `sample_rate` | int | 16000 | 采样率 |
| `hotword` | str | "" | 热词（空格分隔） |
| `use_itn` | bool | true | 逆文本归一化 |
| `hub` | str | "ms" | 仓库：ms/hf |
| `vad_model` | str | None | VAD模型（可选） |
| `punc_model` | str | None | 标点模型（可选） |

## 使用示例

### 示例1：基本识别

```python
from src.asr import FunASRModule
from config import load_config

config = load_config()
asr = FunASRModule(config['asr'])

# 识别单个文件
text = asr.transcribe_file("test.wav")
print(f"识别结果: {text}")
```

### 示例2：批量处理

```python
import glob

# 批量识别多个文件
audio_files = glob.glob("data/audio_input/*.wav")
texts = asr.transcribe_batch(audio_files)

for file, text in zip(audio_files, texts):
    print(f"{file}: {text}")
```

### 示例3：使用不同设备

```python
# CPU模式
config['asr']['device'] = "cpu"
asr = FunASRModule(config['asr'])
```

### 示例4：长音频处理

```python
# 启用VAD处理长音频
config['asr']['vad_model'] = "fsmn-vad"
config['asr']['vad_kwargs'] = {"max_single_segment_time": 60000}
config['asr']['punc_model'] = "ct-punc"

asr = FunASRModule(config['asr'])

# 现在可以处理任意长度的音频
text = asr.transcribe_file("long_audio.wav")  # 可以是几分钟甚至更长的音频
```

### 示例5：专业领域优化

```python
# 医疗领域热词
asr.set_hotword("CT MRI 心电图 血压 血糖")

# 教育领域热词
asr.set_hotword("微积分 线性代数 机器学习 深度学习")

# 识别
text = asr.transcribe_file("lecture.wav")
```

## 性能优化

### 加速建议

1. **使用GPU**
   ```yaml
   asr:
     device: "cuda"  # 使用GPU加速
   ```

2. **批处理**
   ```yaml
   asr:
     batch_size: 4  # 增加批处理大小
   ```

3. **关闭不必要的模块**
   - 如果不处理长音频，可关闭VAD
   - 如果不需要标点，可关闭punc_model

### 内存优化

```yaml
asr:
  # 减小批处理大小
  batch_size: 1
  
  # 调整VAD参数
  vad_kwargs:
    max_single_segment_time: 30000  # 30秒分段
```

## 故障排除

### 问题1：模型下载慢

**现象**：首次运行长时间卡在下载模型

**解决**：
```bash
# 配置ModelScope镜像（国内用户）
export MODELSCOPE_CACHE=/path/to/your/cache

# 或使用HuggingFace
# config.yaml:
#   hub: "hf"
```

### 问题2：显存不足

**现象**：CUDA OOM错误

**解决**：
```yaml
asr:
  device: "cpu"       # 切换到CPU
  batch_size: 1       # 减小批处理
```

### 问题3：识别不准确

**解决**：
1. 检查音频质量（清晰度、采样率）
2. 设置热词
3. 启用VAD和标点恢复
4. 尝试不同模型

### 问题4：FunASR未安装

**现象**：ImportError

**解决**：
```bash
# 方法1：从源码安装（推荐）
cd FunASR
pip install -e .

# 方法2：pip安装
pip install funasr

# 也需要安装modelscope
pip install modelscope
```

### 问题5：音频格式不支持

**解决**：FunASR支持多种格式：
- WAV, MP3, FLAC, M4A, OGG等
- 自动格式转换

## API参考

### FunASRModule类

#### 方法

- `transcribe(audio_input) -> str`: 识别音频
- `transcribe_file(audio_path) -> str`: 识别文件
- `transcribe_array(audio_array, sample_rate) -> str`: 识别数组
- `transcribe_batch(audio_inputs, batch_size) -> List[str]`: 批量识别
- `set_hotword(hotword)`: 设置热词
- `get_model_info() -> dict`: 获取模型信息

### 输入格式

支持的输入：
- 音频文件路径：`"path/to/audio.wav"`
- 音频数组：`np.ndarray`
- 音频数组列表：`[np.ndarray, ...]`
- wav.scp文件（Kaldi格式）

## 性能指标

根据官方文档：

| 模型 | 准确率 | 速度 | 延迟 |
|------|--------|------|------|
| Paraformer-zh | 98%+ | RTF<0.1 | - |
| Paraformer-zh-streaming | 97%+ | - | 600ms |
| SenseVoice | 98%+ | - | - |

*RTF: Real-Time Factor（实时倍数）*

## 进阶功能

### 说话人分离

```python
config['asr']['spk_model'] = "cam++"
asr = FunASRModule(config['asr'])
result = asr.model.generate("audio.wav")
# 结果包含说话人标签
```

### 情感识别

```python
# 使用SenseVoice模型
config['asr']['model_name'] = "iic/SenseVoiceSmall"
asr = FunASRModule(config['asr'])
```

### 语言识别

```python
# SenseVoice支持多语言识别
# 自动识别语种
```

## 参考资料

- **FunASR官方**: https://github.com/alibaba-damo-academy/FunASR
- **ModelScope**: https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition
- **文档**: https://alibaba-damo-academy.github.io/FunASR/

## 许可证

FunASR模型遵循 [Model License Agreement](https://github.com/alibaba-damo-academy/FunASR/blob/main/MODEL_LICENSE)

