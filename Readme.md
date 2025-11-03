# 智能学伴系统 (LearningFriend)

一个基于语音交互的智能学习助手，实现**语音输入 → ASR → LLM → TTS → 语音输出**的完整对话流程。

## ✨ 特性

- 🎤 **高质量语音识别**：使用阿里达摩院的 FunASR，支持中文语音识别
- 🤖 **智能对话**：集成硅基流动的 DeepSeek-V3 大语言模型
- 🔊 **语音合成**：支持 IndexTTS2 官方模型和复现模型
- 🔄 **完整对话流程**：自动化的端到端语音对话处理
- 📝 **对话历史管理**：支持多轮对话上下文
- ⚙️ **灵活配置**：通过 YAML 文件轻松配置各项参数

## 🏗️ 系统架构

```
语音输入(wav/mp3) → FunASR(语音识别) → DeepSeek-V3(文本生成) → IndexTTS2(语音合成) → 语音输出(wav)
                      ↓                      ↓                         ↓
                  中文文本              智能回复文本                音频波形
```

### 核心模块

1. **ASR模块** (`src/asr/`): FunASR 中文语音识别 - [详细文档](src/asr/README.md)
2. **LLM模块** (`src/llm/`): 硅基流动 DeepSeek-V3 API 接口 - [详细文档](src/llm/README.md)
3. **TTS模块** (`src/tts/`): IndexTTS2 语音合成 - [详细文档](src/tts/README.md)
4. **Pipeline模块** (`src/pipeline/`): 对话流程控制器

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA（推荐，用于GPU加速）或 CPU
- 硅基流动 API Key（获取地址：https://siliconflow.cn/）

> 💡 **在 AutoDL 上部署？** 请查看 [AutoDL 部署指南](docs/DEPLOY_AUTODL.md)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository_url>
   cd LearningFriend
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **安装FunASR**（如果未安装）
   ```bash
   cd FunASR
   pip install -e .
   cd ..
   ```

4. **配置API Key**
   
   **首次使用**：从示例文件创建配置文件
   ```bash
   cp config/config.yaml.example config/config.yaml
   ```
   
   然后编辑 `config/config.yaml`，填入你的 API Key：
   ```yaml
   llm:
     provider: "deepseek"
     deepseek:
       api_key: "sk-your-api-key-here"  # ⚠️ 替换为你的硅基流动API Key
       base_url: "https://api.siliconflow.cn/v1"
       model: "DeepSeek/DeepSeek-V3"
   ```
   
   **重要**：
   - ✅ `config.yaml` 已添加到 `.gitignore`，不会被提交到Git
   - ⚠️ **如果 `config.yaml` 已被 Git 跟踪**：请查看 [SECURITY.md](SECURITY.md) 了解如何安全处理
   
   获取API Key：访问 https://siliconflow.cn/

5. **运行测试**
   ```bash
   python test_pipeline.py
   ```
   
   这个测试会：
   - ✓ 测试ASR模块（FunASR）
   - ✓ 测试LLM模块（DeepSeek-V3）
   - ✓ 测试TTS模块（IndexTTS2占位）
   - ✓ 测试完整对话流程
   
   **首次运行提示**：
   - FunASR模型会自动从ModelScope下载（约1-2GB）
   - 如果配置了API Key，会测试LLM对话
   - TTS当前为占位模式，会返回静音音频

6. **开始使用**
   ```bash
   # 交互式模式
   python main.py --mode interactive
   
   # 单文件模式
   python main.py --mode single --input your_audio.wav
   
   # 批处理模式
   python main.py --mode batch --input audio_directory/
   ```

## 📁 项目结构

```
LearningFriend/
├── README.md                    # 项目说明（本文件）
├── requirements.txt             # Python依赖
├── main.py                      # 主程序入口
├── test_pipeline.py             # 端到端测试脚本
├── example_simple.py            # 简单示例
├── config/
│   ├── __init__.py
│   └── config.yaml              # 配置文件
├── src/
│   ├── __init__.py
│   ├── asr/                     # ASR模块
│   │   ├── README.md
│   │   └── funasr_module.py
│   ├── llm/                     # LLM模块
│   │   ├── README.md
│   │   └── llm_interface.py
│   ├── tts/                     # TTS模块
│   │   ├── README.md
│   │   └── indextts_module.py
│   └── pipeline/                # 流程控制
│       └── conversation.py
├── models/                      # 模型文件目录
│   ├── funasr/
│   └── indextts2/
├── data/                        # 数据目录
│   ├── audio_input/             # 输入音频
│   ├── audio_output/            # 输出音频
│   └── logs/                    # 日志文件
├── tests/                       # 单元测试
│   ├── test_asr.py
│   ├── test_llm.py
│   └── test_tts.py
└── scripts/                     # 工具脚本
    ├── download_models.sh
    └── setup_funasr.sh
```

## ⚙️ 核心配置

### ASR配置

```yaml
asr:
  model_name: "paraformer-zh"    # FunASR模型
  device: "cuda"                 # 或 "cpu"
  sample_rate: 16000            # 采样率
  hotword: ""                    # 热词优化（空格分隔）
  vad_model: "fsmn-vad"          # VAD模型（可选）
  punc_model: "ct-punc"          # 标点恢复（可选）
```

### LLM配置

```yaml
llm:
  provider: "deepseek"
  deepseek:
    api_key: "sk-..."            # 硅基流动API Key（必填）
    base_url: "https://api.siliconflow.cn/v1"
    model: "DeepSeek/DeepSeek-V3"
    temperature: 0.7            # 创造性（0.0-2.0）
    max_tokens: 2000           # 回复长度
  system_prompt: |              # 系统提示词
    你是一个友好、耐心的智能学伴助手...
```

### TTS配置

```yaml
tts:
  use_official: true            # 使用官方模型（推荐）
  device: "cuda"
  speed: 1.0                    # 语速（0.5-2.0）
  pitch: 1.0                    # 音高（0.5-2.0）
  sample_rate: 22050
```

完整配置说明请查看各模块的 README.md 文件。

## 📊 使用示例

### Python代码

```python
from config import load_config
from src.pipeline import ConversationPipeline

# 加载配置
config = load_config('config/config.yaml')

# 初始化系统
pipeline = ConversationPipeline(config)

# 处理音频文件
result = pipeline.process_audio_file('audio.wav')

if result['success']:
    print(f"用户: {result['asr_text']}")
    print(f"助手: {result['llm_response']}")
    print(f"音频: {result['output_audio_path']}")
```

### 命令行

```bash
# 交互式对话
python main.py --mode interactive

# 处理单个文件
python main.py --mode single --input test.wav

# 批量处理
python main.py --mode batch --input audio_folder/
```

### 单独使用各模块

#### ASR模块

```python
from src.asr import FunASRModule
from config import load_config

config = load_config()
asr = FunASRModule(config['asr'])

# 识别文件
text = asr.transcribe_file('audio.wav')

# 设置热词
asr.set_hotword("机器学习 深度学习")
```

详细用法请查看 [src/asr/README.md](src/asr/README.md)

#### LLM模块

```python
from src.llm import LLMInterface
from config import load_config

config = load_config()
llm = LLMInterface(config['llm'])

# 发送消息
response = llm.chat("你好")

# 查看历史
history = llm.get_history()

# 清空历史
llm.clear_history()
```

详细用法请查看 [src/llm/README.md](src/llm/README.md)

#### TTS模块

```python
from src.tts import create_tts_module
from config import load_config

config = load_config()
tts = create_tts_module(config['tts'])

# 合成语音
audio = tts.synthesize("你好，世界！")

# 保存到文件
tts.synthesize_to_file("你好", "output.wav")
```

详细用法请查看 [src/tts/README.md](src/tts/README.md)

## 🧪 测试

```bash
# 端到端测试
python test_pipeline.py

# 单元测试
pytest tests/ -v
pytest tests/test_asr.py -v
pytest tests/test_llm.py -v
pytest tests/test_tts.py -v
```

## 🔧 常见问题

### FunASR模型下载慢？

**现象**：首次运行长时间卡在下载模型

**解决**：
```bash
# 方法1：配置ModelScope镜像（国内用户推荐）
export MODELSCOPE_CACHE=/path/to/your/cache

# 方法2：使用CPU（更慢但稳定）
# 在 config/config.yaml 中设置:
asr:
  device: "cpu"
```

### GPU不可用或内存不足？

**现象**：CUDA相关错误或OOM错误

**解决**：
```yaml
# 在 config/config.yaml 中设置为CPU:
asr:
  device: "cpu"
tts:
  device: "cpu"
```

### LLM API调用失败？

**现象**：错误信息包含"401", "403"或"Invalid API Key"

**解决**：
```yaml
# 1. 检查API Key是否正确填写
llm:
  deepseek:
    api_key: "sk-..."  # 确保完整且正确

# 2. 检查账户余额
# 登录 https://siliconflow.cn/ 查看余额

# 3. 检查网络连接
# 确保可以访问 https://api.siliconflow.cn/

# 4. 查看详细日志
cat data/logs/system.log
```

### 音频格式不支持？

**现象**：ASR识别失败或无法读取音频

**解决**：
```bash
# 使用ffmpeg转换音频格式
# 转换为推荐格式（16kHz, 单声道, WAV）
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

### IndexTTS2返回静音？

**现象**：输出的音频是静音

**解决**：
- 如果使用官方模型，检查是否正确安装和下载模型
- 如果使用复现模型，需要等待官方发布预训练权重
- 查看 `src/tts/README.md` 了解详细集成步骤

### 导入错误（ModuleNotFoundError）？

**现象**：找不到模块或包

**解决**：
```bash
# 确保在项目根目录运行
cd /path/to/LearningFriend

# 重新安装依赖
pip install -r requirements.txt

# 如果使用FunASR，需要额外安装
cd FunASR
pip install -e .
cd ..
```

## 🛠️ 开发

### 添加新的LLM提供商

1. 在 `config/config.yaml` 中添加配置
2. 在 `src/llm/llm_interface.py` 中实现初始化
3. 测试兼容性

### 自定义TTS

1. 替换 `src/tts/indextts_module.py` 或使用官方模型包装器
2. 实现 `synthesize()` 方法
3. 更新配置文件

### 贡献

欢迎提交 Issue 和 Pull Request！

## 📚 详细文档

- **[src/asr/README.md](src/asr/README.md)** - ASR模块完整文档
- **[src/llm/README.md](src/llm/README.md)** - LLM模块完整文档
- **[src/tts/README.md](src/tts/README.md)** - TTS模块完整文档

## 📝 注意事项

1. **API Key安全**：
   - ✅ `config/config.yaml` 已添加到 `.gitignore`，不会被提交到Git
   - ✅ 使用 `config/config.yaml.example` 作为模板，不包含敏感信息
   - ✅ 首次使用请复制示例文件：`cp config/config.yaml.example config/config.yaml`
   - ⚠️ 如果之前已提交过 `config.yaml`，请立即撤销并重新生成API Key
2. **模型文件**：模型较大，建议使用`.gitignore`排除
3. **GPU内存**：如不足可切换到CPU
4. **TTS模块**：官方模型需要下载约5.9GB，首次使用需要时间

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里达摩院语音实验室
- [硅基流动](https://siliconflow.cn/) - DeepSeek-V3 模型服务
- [DeepSeek](https://www.deepseek.com/) - DeepSeek AI
- [IndexTTS2](https://github.com/index-tts/index-tts) - 高质量语音合成系统

## 📄 许可证

MIT License

## 📧 联系方式

- 提交问题：GitHub Issues
- 讨论建议：GitHub Discussions

---

**Happy Learning! 📚✨**
