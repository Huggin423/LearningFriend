# API 服务说明文档

## 概述

本项目提供了完整的语音识别（ASR）和大语言模型（LLM）API服务，支持独立部署或统一部署。

## 架构设计

```
src/api/
├── main.py           # 主API服务（整合所有服务）
├── asr_api.py       # ASR独立服务
├── llm_api.py       # LLM独立服务
├── models.py        # 数据模型定义
├── utils.py         # 工具函数
└── README.md        # 本文档
```

## 部署方式

### 方式1: 统一部署（推荐）

启动主服务，提供所有功能的统一入口：

```bash
# 启动主服务（默认端口 8000）
python -m src.api.main

# 或指定端口
API_HOST=0.0.0.0 API_PORT=8000 python -m src.api.main
```

### 方式2: 独立部署

分别启动ASR和LLM服务：

```bash
# 启动ASR服务（默认端口 8001）
python -m src.api.asr_api

# 启动LLM服务（默认端口 8002）
python -m src.api.llm_api
```

## API 接口文档

### 主服务 (main.py) - 端口 8000

#### 1. 健康检查

```http
GET /health
```

响应示例：
```json
{
  "status": "healthy",
  "services": {
    "asr": true,
    "llm": true
  },
  "version": "1.0.0"
}
```

#### 2. 加载服务

```http
POST /load/asr        # 加载ASR模型
POST /load/llm        # 初始化LLM客户端
POST /load/all        # 加载所有服务
```

#### 3. ASR 语音识别

```http
POST /asr/transcribe
```

请求体：
```json
{
  "audio_path": "/path/to/audio.wav",  // 方式1: 本地文件路径
  "audio_base64": "base64_string",     // 方式2: Base64编码音频
  "language": "zh",                     // 语言代码
  "use_itn": true                       // 是否使用逆文本归一化
}
```

响应示例：
```json
{
  "success": true,
  "text": "这是识别出的文本内容",
  "message": null,
  "duration": 0.85
}
```

#### 4. LLM 对话

```http
POST /llm/chat
```

请求体：
```json
{
  "message": "你好，请介绍一下自己",
  "use_history": true,                 // 是否使用对话历史
  "system_prompt": "你是一个友好的助手", // 可选：临时系统提示词
  "temperature": 0.7,                   // 可选：温度参数
  "max_tokens": 2000                    // 可选：最大token数
}
```

响应示例：
```json
{
  "success": true,
  "reply": "你好！我是一个AI助手...",
  "message": null,
  "duration": 1.23
}
```

#### 5. 综合对话流程

```http
POST /conversation
```

这是一个便捷接口，自动执行：音频识别 → LLM对话 → 返回文字回复

请求体（与 ASR 相同）：
```json
{
  "audio_path": "/path/to/audio.wav",
  "language": "zh",
  "use_itn": true
}
```

响应示例：
```json
{
  "success": true,
  "reply": "我很好，谢谢！你呢？",
  "message": "用户说: 你好，最近怎么样",
  "duration": 2.15
}
```

#### 6. 清空对话历史

```http
POST /llm/history/clear
```

### ASR 独立服务 (asr_api.py) - 端口 8001

提供完整的ASR功能，包括：

- `POST /load` - 加载模型
- `POST /transcribe` - 单个音频识别
- `POST /transcribe/batch` - 批量识别
- `GET /model/info` - 获取模型信息
- `GET /health` - 健康检查

### LLM 独立服务 (llm_api.py) - 端口 8002

提供完整的LLM功能，包括：

- `POST /load` - 初始化客户端
- `POST /chat` - 对话
- `GET /history` - 获取对话历史
- `POST /history/clear` - 清空历史
- `POST /history/trim` - 修剪历史
- `POST /system_prompt` - 设置系统提示词
- `POST /switch_provider` - 切换提供商
- `GET /model/info` - 获取模型信息
- `GET /health` - 健康检查

## 使用示例

### Python 客户端示例

```python
import requests
import base64

# API基础URL
BASE_URL = "http://localhost:8000"

# 1. 加载所有服务
response = requests.post(f"{BASE_URL}/load/all")
print(response.json())

# 2. 语音识别
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{BASE_URL}/asr/transcribe",
    json={
        "audio_base64": audio_base64,
        "language": "zh",
        "use_itn": True
    }
)
print(response.json())

# 3. LLM对话
response = requests.post(
    f"{BASE_URL}/llm/chat",
    json={
        "message": "你好，请介绍一下自己",
        "use_history": True
    }
)
print(response.json())

# 4. 综合对话流程（语音输入 -> 文字回复）
response = requests.post(
    f"{BASE_URL}/conversation",
    json={
        "audio_base64": audio_base64,
        "language": "zh"
    }
)
print(response.json())
```

### cURL 示例

```bash
# 加载服务
curl -X POST http://localhost:8000/load/all

# ASR识别（使用本地文件）
curl -X POST http://localhost:8000/asr/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/path/to/audio.wav",
    "language": "zh"
  }'

# LLM对话
curl -X POST http://localhost:8000/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好",
    "use_history": true
  }'

# 健康检查
curl http://localhost:8000/health
```

## 配置说明

API服务使用项目的配置文件 `config/config.yaml`：

```yaml
asr:
  model_name: "paraformer-zh"
  device: "cuda"
  batch_size: 1
  sample_rate: 16000

llm:
  provider: "deepseek"  # 或 "qwen"
  deepseek:
    api_key: "your_api_key"
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    temperature: 0.7
    max_tokens: 2000
```

## 环境变量

可以通过环境变量配置服务端口：

```bash
# 主服务
export API_HOST=0.0.0.0
export API_PORT=8000

# ASR服务
export ASR_API_HOST=0.0.0.0
export ASR_API_PORT=8001

# LLM服务
export LLM_API_HOST=0.0.0.0
export LLM_API_PORT=8002
```

## 交互式API文档

启动服务后，访问以下地址查看自动生成的API文档：

- 主服务: http://localhost:8000/docs
- ASR服务: http://localhost:8001/docs
- LLM服务: http://localhost:8002/docs

## 错误处理

所有API接口都返回统一的错误格式：

```json
{
  "detail": "错误描述信息"
}
```

HTTP状态码：
- `200` - 成功
- `400` - 请求参数错误
- `500` - 服务器错误

## 性能优化建议

1. **预加载模型**：服务启动后立即调用 `/load/all`
2. **批量处理**：使用 `/transcribe/batch` 进行批量ASR识别
3. **历史管理**：定期调用 `/history/trim` 修剪对话历史
4. **音频格式**：推荐使用16kHz采样率的WAV格式
5. **并发控制**：根据GPU显存调整并发请求数

## 依赖项

```bash
pip install fastapi uvicorn pydantic pyyaml
```

## 许可证

MIT License
