# API接口交付文档

## 📋 项目概述

本项目为 LearningFriend 提供了完整的 API 服务接口，包括：
- **ASR (语音识别)**: 基于 FunASR 的语音转文字服务
- **LLM (对话)**: 基于 DeepSeek/Qwen 的智能对话服务
- **综合服务**: 语音输入 -> 识别 -> 对话 -> 文字回复的完整流程

## 📁 文件结构

```
LearningFriend/
├── src/
│   ├── api/                    # 新增的API模块 ✨
│   │   ├── __init__.py        # 模块初始化
│   │   ├── main.py            # 主API服务（统一入口）
│   │   ├── asr_api.py         # ASR独立服务
│   │   ├── llm_api.py         # LLM独立服务
│   │   ├── models.py          # Pydantic数据模型
│   │   ├── utils.py           # 工具函数
│   │   ├── test_client.py     # 测试客户端
│   │   └── README.md          # API文档
│   ├── asr/                    # 现有ASR模块
│   │   └── funasr_module.py
│   └── llm/                    # 现有LLM模块
│       └── llm_interface.py
├── scripts/
│   ├── start_api.sh           # Linux/Mac启动脚本 ✨
│   └── start_api.bat          # Windows启动脚本 ✨
├── requirements-api.txt        # API依赖清单 ✨
└── config/
    └── config.yaml            # 配置文件（需要配置）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装API依赖
pip install -r requirements-api.txt
```

主要依赖：
- `fastapi` - Web框架
- `uvicorn` - ASGI服务器
- `pydantic` - 数据验证
- `pyyaml` - 配置文件解析

### 2. 配置文件

确保 `config/config.yaml` 已正确配置：

```yaml
asr:
  model_name: "paraformer-zh"
  device: "cuda"
  batch_size: 1
  sample_rate: 16000

llm:
  provider: "deepseek"  # 或 "qwen"
  system_prompt: "你是一个友好的智能助手。"
  
  deepseek:
    api_key: "your_api_key_here"  # ⚠️ 必填
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    temperature: 0.7
    max_tokens: 2000
```

### 3. 启动服务

#### 方式A: 使用启动脚本（推荐）

**Windows:**
```bash
.\scripts\start_api.bat
```

**Linux/Mac:**
```bash
bash scripts/start_api.sh
```

#### 方式B: 手动启动

**启动主服务（推荐）:**
```bash
python -m src.api.main
# 访问: http://localhost:8000
# API文档: http://localhost:8000/docs
```

**或分别启动独立服务:**
```bash
# ASR服务
python -m src.api.asr_api
# 访问: http://localhost:8001

# LLM服务
python -m src.api.llm_api
# 访问: http://localhost:8002
```

### 4. 测试服务

```bash
# 健康检查
curl http://localhost:8000/health

# 运行测试客户端
python -m src.api.test_client
```

## 📖 API接口文档

### 核心接口

#### 1. 加载服务
```http
POST /load/all          # 加载所有服务（推荐首次调用）
POST /load/asr          # 仅加载ASR
POST /load/llm          # 仅加载LLM
```

#### 2. ASR 语音识别
```http
POST /asr/transcribe
Content-Type: application/json

{
  "audio_path": "/path/to/audio.wav",  // 或使用 audio_base64
  "language": "zh",
  "use_itn": true
}
```

响应：
```json
{
  "success": true,
  "text": "识别出的文本内容",
  "duration": 0.85
}
```

#### 3. LLM 对话
```http
POST /llm/chat
Content-Type: application/json

{
  "message": "你好，请介绍一下自己",
  "use_history": true,
  "temperature": 0.7
}
```

响应：
```json
{
  "success": true,
  "reply": "你好！我是一个AI助手...",
  "duration": 1.23
}
```

#### 4. 综合对话流程 ⭐
```http
POST /conversation
Content-Type: application/json

{
  "audio_path": "/path/to/audio.wav",
  "language": "zh"
}
```

自动执行：音频识别 → LLM对话 → 返回文字回复

响应：
```json
{
  "success": true,
  "reply": "我很好，谢谢！",
  "message": "用户说: 你好，最近怎么样",
  "duration": 2.15
}
```

### 完整接口列表

详见：`src/api/README.md`

## 💻 使用示例

### Python客户端

```python
import requests
import base64

BASE_URL = "http://localhost:8000"

# 1. 加载服务
requests.post(f"{BASE_URL}/load/all")

# 2. ASR识别
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{BASE_URL}/asr/transcribe",
    json={"audio_base64": audio_base64, "language": "zh"}
)
print(response.json())

# 3. LLM对话
response = requests.post(
    f"{BASE_URL}/llm/chat",
    json={"message": "你好", "use_history": True}
)
print(response.json())

# 4. 综合对话流程
response = requests.post(
    f"{BASE_URL}/conversation",
    json={"audio_base64": audio_base64}
)
print(response.json())
```

### cURL示例

```bash
# 加载服务
curl -X POST http://localhost:8000/load/all

# ASR识别
curl -X POST http://localhost:8000/asr/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/to/audio.wav", "language": "zh"}'

# LLM对话
curl -X POST http://localhost:8000/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "use_history": true}'
```

### JavaScript示例

```javascript
const BASE_URL = "http://localhost:8000";

// 加载服务
await fetch(`${BASE_URL}/load/all`, { method: 'POST' });

// LLM对话
const response = await fetch(`${BASE_URL}/llm/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "你好",
    use_history: true
  })
});

const data = await response.json();
console.log(data.reply);
```

## 🔧 高级功能

### 1. 批量ASR识别

```python
response = requests.post(
    f"{BASE_URL}/asr/transcribe/batch",
    json={
        "audio_paths": [
            "/path/to/audio1.wav",
            "/path/to/audio2.wav",
            "/path/to/audio3.wav"
        ],
        "batch_size": 4
    }
)
```

### 2. 自定义系统提示词

```python
# 临时使用不同的系统提示词
response = requests.post(
    f"{BASE_URL}/llm/chat",
    json={
        "message": "什么是人工智能？",
        "use_history": False,
        "system_prompt": "你是一位资深的AI专家，用专业术语解释问题"
    }
)
```

### 3. 对话历史管理

```python
# 获取对话历史
history = requests.get(f"{BASE_URL}/llm/history").json()

# 清空对话历史
requests.post(f"{BASE_URL}/llm/history/clear")

# 修剪对话历史（保留最近10轮）
requests.post(f"{BASE_URL}/llm/history/trim?max_turns=10")
```

### 4. 切换LLM提供商

```python
# 切换到Qwen
requests.post(
    f"{BASE_URL}/llm/switch_provider?provider=qwen"
)
```

## 📊 API文档

启动服务后访问：
- **主服务**: http://localhost:8000/docs
- **ASR服务**: http://localhost:8001/docs
- **LLM服务**: http://localhost:8002/docs

提供交互式API文档（Swagger UI）。

## 🔍 测试工具

### 使用测试客户端

```bash
python -m src.api.test_client
```

提供交互式菜单，包括：
1. 基础使用示例（LLM对话）
2. ASR语音识别示例
3. 综合对话流程示例
4. 自定义系统提示词示例

### 使用curl测试

```bash
# 健康检查
curl http://localhost:8000/health

# 查看服务状态
curl http://localhost:8000/health | python -m json.tool
```

## ⚙️ 配置选项

### 环境变量

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

# 日志级别
export LOG_LEVEL=INFO
```

### CORS配置

如果需要从前端调用API，已经配置了CORS：

```python
# src/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🚨 常见问题

### 1. 模型加载失败

**问题**: `模型未加载，请先调用 /load 接口`

**解决**: 
```bash
curl -X POST http://localhost:8000/load/all
```

### 2. LLM API Key错误

**问题**: `DeepSeek API Key未配置`

**解决**: 编辑 `config/config.yaml`，填入正确的API Key

### 3. 音频文件格式不支持

**问题**: `音频文件不存在或格式不支持`

**解决**: 
- 支持格式: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`
- 推荐使用 16kHz 采样率的 WAV 格式

### 4. 端口被占用

**问题**: `Address already in use`

**解决**: 
```bash
# 更改端口
API_PORT=8080 python -m src.api.main
```

## 📈 性能优化

1. **预加载模型**: 服务启动后立即调用 `/load/all`
2. **批量处理**: 使用批量接口处理多个音频
3. **历史管理**: 定期清理或修剪对话历史
4. **并发控制**: 根据GPU显存调整并发数

## 🔒 安全建议

1. **生产环境**: 限制CORS域名
2. **API Key**: 使用环境变量管理敏感信息
3. **认证**: 可添加API Key或Token认证
4. **HTTPS**: 生产环境使用HTTPS

## 📝 部署清单

- [x] 创建API模块 (`src/api/`)
- [x] 实现ASR接口 (`asr_api.py`)
- [x] 实现LLM接口 (`llm_api.py`)
- [x] 实现主服务 (`main.py`)
- [x] 定义数据模型 (`models.py`)
- [x] 实现工具函数 (`utils.py`)
- [x] 编写API文档 (`README.md`)
- [x] 创建测试客户端 (`test_client.py`)
- [x] 准备依赖清单 (`requirements-api.txt`)
- [x] 创建启动脚本 (`start_api.sh/bat`)
- [x] 编写交付文档（本文件）

## 🎯 下一步

1. 根据实际音频测试ASR识别效果
2. 配置LLM的API Key并测试对话功能
3. 根据需求调整系统提示词
4. 集成TTS服务（如需要）
5. 添加认证机制（如需要）
6. 部署到生产环境

## 📞 技术支持

如有问题，请参考：
- API文档: `src/api/README.md`
- 测试客户端: `src/api/test_client.py`
- 配置示例: `config/config.yaml.example`

---

**版本**: 1.0.0  
**日期**: 2025-11-10  
**作者**: GitHub Copilot
