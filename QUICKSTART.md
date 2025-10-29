# 快速入门指南

本指南将帮助你在5分钟内运行智能学伴系统。

## 🎯 最小化安装步骤

### 步骤1：安装Python依赖

```bash
pip install -r requirements.txt
```

### 步骤2：配置API Key

编辑 `config/config.yaml`，填入你的API Key：

```yaml
llm:
  deepseek:
    api_key: "sk-your-api-key-here"  # 替换为你的API Key
```

**获取API Key**：
- DeepSeek: https://platform.deepseek.com/
- Qwen: https://dashscope.console.aliyun.com/

### 步骤3：首次运行（模型会自动下载）

```bash
python main.py --mode interactive
```

第一次运行时，FunASR模型会自动从 ModelScope 下载（约500MB-1GB）。

### 步骤4：开始对话

准备一个测试音频文件（`.wav`, `.mp3` 等格式），然后：

1. 运行系统后，输入音频文件路径
2. 系统会自动完成：语音识别 → LLM对话 → 语音合成
3. 查看终端输出和生成的语音文件

## 📝 测试示例

### 准备测试音频

如果你没有音频文件，可以：

1. **录制音频**：使用手机或电脑录一段中文语音（如"你好，请介绍一下自己"）
2. **转换格式**：确保是 `.wav` 格式（或使用 ffmpeg 转换）
3. **放到指定目录**：
   ```bash
   cp your_audio.wav data/audio_input/test.wav
   ```

### 运行测试

```bash
# 交互式模式
python main.py --mode interactive
# 然后输入: data/audio_input/test.wav

# 或者直接单文件模式
python main.py --mode single --input data/audio_input/test.wav
```

### 预期输出

```
====================================
智能学伴系统 - 单文件模式
====================================
输入文件: data/audio_input/test.wav
====================================

处理中...

────────────────────────────────────
👤 用户: 你好，请介绍一下自己
🤖 助手: 你好！我是智能学伴助手，很高兴认识你...
────────────────────────────────────
🔊 语音已保存: data/audio_output/response_20251029_213000_0001.wav

✓ 处理成功
```

## 🔧 常见问题

### Q: FunASR模型下载慢怎么办？

A: 模型会缓存在 `~/.cache/modelscope/`，只需下载一次。如果网络问题，可以：
```bash
# 使用镜像（如果可用）
export MODELSCOPE_CACHE=/path/to/your/cache
```

### Q: GPU不可用怎么办？

A: 修改 `config/config.yaml`：
```yaml
asr:
  device: "cpu"
tts:
  device: "cpu"
```

### Q: API调用失败？

A: 检查：
1. API Key 是否正确填写
2. 是否有网络连接
3. 是否有足够的API额度

查看日志：
```bash
cat data/logs/system.log
```

### Q: IndexTTS2模型在哪里？

A: 目前TTS模块是预留接口，返回静音音频。等待 IndexTTS2 正式发布后：
1. 下载模型文件
2. 放到 `models/indextts2/` 目录
3. 更新 `src/tts/indextts_module.py` 中的加载逻辑

## 🚀 进阶使用

### 批处理多个音频

```bash
# 将多个音频文件放到一个目录
mkdir data/audio_input/batch/
cp *.wav data/audio_input/batch/

# 批处理
python main.py --mode batch --input data/audio_input/batch/
```

### 调整参数

编辑 `config/config.yaml`：

```yaml
# 更换LLM提供商
llm:
  provider: "qwen"  # 改为qwen

# 调整语速和音高
tts:
  speed: 1.2  # 加快20%
  pitch: 1.1  # 提高音高
  
# 调整对话历史长度
conversation:
  max_history: 20  # 保留20轮对话
```

### Jupyter Notebook

```bash
# 启动Jupyter
jupyter notebook

# 打开 notebooks/demo.ipynb
```

## 📚 下一步

- 阅读完整文档：[README.md](README.md)
- 查看配置说明：[config/config.yaml](config/config.yaml)
- 运行测试：`pytest tests/ -v`
- 查看演示：`notebooks/demo.ipynb`

## 💡 提示

1. **保存音频**：所有生成的语音都保存在 `data/audio_output/`
2. **查看日志**：详细日志在 `data/logs/system.log`
3. **重置对话**：交互模式下输入 `reset` 可重置对话历史
4. **退出系统**：交互模式下输入 `quit` 退出

---

祝你使用愉快！如有问题，请查看 [README.md](README.md) 或提交 Issue。

