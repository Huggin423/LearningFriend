# 智能学伴系统

一个基于语音交互的智能学习助手，支持**语音输入 → 语音识别(ASR) → 大语言模型(LLM) → 语音合成(TTS) → 语音输出**的完整对话流程。

## ✨ 特性

- 🎤 **高质量语音识别**：使用阿里达摩院的 FunASR，支持中文语音识别
- 🤖 **智能对话**：支持 DeepSeek 和 Qwen 等商业 LLM API
- 🔊 **自然语音合成**：基于 IndexTTS2 的高质量语音合成
- 🔄 **完整对话流程**：自动化的端到端语音对话处理
- 📝 **对话历史管理**：支持多轮对话上下文
- ⚙️ **灵活配置**：通过 YAML 文件轻松配置各项参数

## 📋 系统架构

```
语音输入 → FunASR(语音识别) → LLM(文本生成) → IndexTTS2(语音合成) → 语音输出
```

### 核心模块

1. **ASR模块** (`src/asr/`): FunASR 语音识别
2. **LLM模块** (`src/llm/`): DeepSeek/Qwen API 接口
3. **TTS模块** (`src/tts/`): IndexTTS2 语音合成
4. **Pipeline模块** (`src/pipeline/`): 对话流程控制

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA（推荐，用于GPU加速）
- Git

### 2. 安装依赖

```bash
# 克隆项目（如果还没有）
git clone <repository_url>
cd LearningFriend

# 安装Python依赖
pip install -r requirements.txt

# 设置FunASR环境（如果已克隆FunASR）
bash scripts/setup_funasr.sh
```

### 3. 配置系统

编辑 `config/config.yaml`，填入你的 API Key：

```yaml
llm:
  provider: "deepseek"  # 或 "qwen"
  
  deepseek:
    api_key: "your-api-key-here"  # 填入你的API Key
```

### 4. 下载模型

```bash
# 运行模型下载脚本
bash scripts/download_models.sh
```

**注意**：
- FunASR 模型会在首次运行时自动下载
- IndexTTS2 模型需要手动下载并放置到 `models/indextts2/` 目录

### 5. 运行系统

#### 交互式模式（推荐入门）

```bash
python main.py --mode interactive
```

按提示输入音频文件路径进行对话。

#### 单文件模式

```bash
python main.py --mode single --input path/to/audio.wav
```

#### 批处理模式

```bash
python main.py --mode batch --input path/to/audio_directory/
```

## 📁 项目结构

```
LearningFriend/
├── README.md                    # 项目说明
├── requirements.txt             # Python依赖
├── main.py                      # 主程序入口
├── config/
│   ├── __init__.py
│   └── config.yaml              # 配置文件
├── src/
│   ├── __init__.py
│   ├── asr/                     # ASR模块
│   │   ├── __init__.py
│   │   └── funasr_module.py
│   ├── llm/                     # LLM模块
│   │   ├── __init__.py
│   │   └── llm_interface.py
│   ├── tts/                     # TTS模块
│   │   ├── __init__.py
│   │   └── indextts_module.py
│   └── pipeline/                # 流程控制
│       ├── __init__.py
│       └── conversation.py
├── models/                      # 模型文件
│   ├── funasr/
│   └── indextts2/
├── data/                        # 数据目录
│   ├── audio_input/             # 输入音频
│   ├── audio_output/            # 输出音频
│   └── logs/                    # 日志文件
├── tests/                       # 测试文件
│   ├── test_asr.py
│   ├── test_llm.py
│   └── test_tts.py
├── scripts/                     # 脚本工具
    ├── download_models.sh
    └── setup_funasr.sh

```

## ⚙️ 配置说明

### ASR配置

```yaml
asr:
  model_name: "paraformer-zh"    # 模型名称
  device: "cuda"                 # cuda或cpu
  sample_rate: 16000             # 采样率
  use_itn: true                  # 逆文本归一化
```

### LLM配置

```yaml
llm:
  provider: "deepseek"           # deepseek或qwen
  deepseek:
    api_key: "your-key"
    model: "deepseek-chat"
    temperature: 0.7
```

### TTS配置

```yaml
tts:
  model_path: "models/indextts2"
  device: "cuda"
  speaker_id: 0                  # 音色ID
  speed: 1.0                     # 语速
  pitch: 1.0                     # 音高
```

## 🧪 测试

运行测试套件：

```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/test_asr.py -v
pytest tests/test_llm.py -v
pytest tests/test_tts.py -v
```

## 📊 使用示例

### Python代码示例

```python
from config import load_config
from src.pipeline import ConversationPipeline

# 加载配置
config = load_config('config/config.yaml')

# 初始化对话流程
pipeline = ConversationPipeline(config)

# 处理音频文件
result = pipeline.process_audio_file('path/to/audio.wav')

if result['success']:
    print(f"用户: {result['asr_text']}")
    print(f"助手: {result['llm_response']}")
    print(f"输出音频: {result['output_audio_path']}")
```

## 🔧 开发指南

### 添加新的LLM提供商

1. 在 `config/config.yaml` 中添加新的提供商配置
2. 在 `src/llm/llm_interface.py` 中添加初始化逻辑
3. 更新 `switch_provider` 方法

### 自定义TTS模型

1. 将 IndexTTS2 模型替换为你的 TTS 实现
2. 更新 `src/tts/indextts_module.py` 中的模型加载和推理逻辑
3. 调整配置文件中的 TTS 参数

## 📝 注意事项

1. **API Key 安全**：不要将包含 API Key 的配置文件提交到 Git
2. **模型文件**：模型文件较大，建议使用 `.gitignore` 排除
3. **GPU内存**：如果GPU内存不足，可以在配置中设置 `device: "cpu"`
4. **IndexTTS2**：目前为预留接口，需要根据实际实现调整代码

## 🐛 故障排除

### FunASR 模型下载失败

```bash
# 手动设置 ModelScope 镜像
export MODELSCOPE_CACHE=~/.cache/modelscope
```

### CUDA 相关错误

```yaml
# 在配置文件中使用CPU
asr:
  device: "cpu"
tts:
  device: "cpu"
```

### API调用失败

- 检查 API Key 是否正确
- 检查网络连接
- 查看 `data/logs/system.log` 获取详细错误信息

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里达摩院语音实验室
- [DeepSeek](https://www.deepseek.com/) - DeepSeek AI
- [Qwen](https://qwen.aliyun.com/) - 阿里云通义千问
- IndexTTS2 - 高质量语音合成系统

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**Happy Learning! 📚✨**
