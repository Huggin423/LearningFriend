# LLM 大语言模型模块

基于 OpenAI 兼容接口的大语言模型模块，支持多种商业 API 提供商。

## ✨ 特性

- ✅ **多提供商支持**：支持 DeepSeek、Qwen 等
- ✅ **对话历史管理**：自动维护多轮对话上下文
- ✅ **系统提示词**：可自定义助手行为和风格
- ✅ **灵活配置**：支持 temperature、max_tokens 等参数
- ✅ **历史修剪**：自动管理对话历史长度
- ✅ **OpenAI 兼容**：使用标准 OpenAI API 接口

## 🚀 快速开始

### 基本使用

```python
from config import load_config
from src.llm import LLMInterface

# 加载配置
config = load_config()
llm_config = config['llm']

# 初始化模块
llm = LLMInterface(llm_config)

# 发送消息
response = llm.chat("你好")
print(response)
```

### 配置说明

在 `config/config.yaml` 中配置：

```yaml
llm:
  provider: "deepseek"  # 选择提供商
  
  deepseek:
    api_key: "sk-your-api-key"  # API Key（必填）
    base_url: "https://api.siliconflow.cn/v1"
    model: "DeepSeek/DeepSeek-V3"
    temperature: 0.7          # 创造性（0.0-2.0）
    max_tokens: 2000          # 最大回复长度
    top_p: 0.95              # 核采样参数
  
  qwen:
    api_key: "sk-your-qwen-key"
    base_url: "https://dashscope.aliyuncs.com/api/v1"
    model: "qwen-turbo"
    temperature: 0.7
    max_tokens: 2000
  
  # 系统提示词
  system_prompt: |
    你是一个友好、耐心的智能学伴助手。
    能够回答各种学习问题，提供清晰准确的解释。
```

## 📖 API 参考

### LLMInterface 类

#### 初始化

```python
llm = LLMInterface(config)
```

**参数**：
- `config`: LLM配置字典，包含provider、api_key等

#### 方法

##### `chat(user_message, use_history=True)`

发送消息并获取回复。

```python
# 使用对话历史
response = llm.chat("你好")

# 不使用历史（单次对话）
response = llm.chat("你好", use_history=False)
```

**参数**：
- `user_message` (str): 用户消息
- `use_history` (bool): 是否使用对话历史，默认True

**返回**：
- `str`: LLM的回复文本

##### `clear_history()`

清空对话历史。

```python
llm.clear_history()
```

##### `get_history()`

获取当前对话历史。

```python
history = llm.get_history()
# 返回: [{"role": "user", "content": "..."}, ...]
```

##### `set_history(history)`

设置对话历史。

```python
history = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
]
llm.set_history(history)
```

##### `trim_history(max_turns=10)`

修剪对话历史，保留最近的N轮对话。

```python
llm.trim_history(max_turns=5)  # 保留最近5轮对话
```

##### `set_system_prompt(prompt)`

设置系统提示词。

```python
llm.set_system_prompt("你是一个专业的数学老师。")
```

##### `switch_provider(provider)`

切换LLM提供商。

```python
llm.switch_provider("qwen")  # 切换到Qwen
```

**支持的提供商**：
- `"deepseek"`: DeepSeek模型（硅基流动）
- `"qwen"`: Qwen模型（阿里云）

## 💡 使用示例

### 示例1：基本对话

```python
from src.llm import LLMInterface
from config import load_config

config = load_config()
llm = LLMInterface(config['llm'])

# 单轮对话
response = llm.chat("什么是机器学习？")
print(response)
```

### 示例2：多轮对话

```python
# 第一轮
response1 = llm.chat("什么是机器学习？")
print(f"助手: {response1}")

# 第二轮（会使用历史上下文）
response2 = llm.chat("能给我举个例子吗？")
print(f"助手: {response2}")
```

### 示例3：不使用历史

```python
# 每次对话都是独立的
response1 = llm.chat("你好", use_history=False)
response2 = llm.chat("再见", use_history=False)
```

### 示例4：自定义系统提示词

```python
# 设置为数学老师
llm.set_system_prompt("你是一个专业的数学老师，用简洁清晰的语言解释概念。")

response = llm.chat("什么是微积分？")
```

### 示例5：管理对话历史

```python
# 获取历史
history = llm.get_history()
print(f"当前有 {len(history)} 条历史记录")

# 修剪历史（保留最近5轮）
llm.trim_history(max_turns=5)

# 清空历史
llm.clear_history()
```

### 示例6：切换提供商

```python
# 从DeepSeek切换到Qwen
llm.switch_provider("qwen")

# 继续使用
response = llm.chat("你好")
```

## ⚙️ 配置参数详解

### DeepSeek配置

```yaml
llm:
  provider: "deepseek"
  
  deepseek:
    api_key: "sk-..."                    # 硅基流动API Key（必填）
    base_url: "https://api.siliconflow.cn/v1"  # API地址
    model: "DeepSeek/DeepSeek-V3"         # 模型名称
    temperature: 0.7                      # 创造性：0.0（确定）到2.0（随机）
    max_tokens: 2000                      # 最大回复长度
    top_p: 0.95                          # 核采样参数：0.0-1.0
```

**参数说明**：

| 参数 | 类型 | 说明 | 推荐值 |
|------|------|------|--------|
| `api_key` | str | API密钥（必填） | - |
| `base_url` | str | API基础地址 | `https://api.siliconflow.cn/v1` |
| `model` | str | 模型名称 | `DeepSeek/DeepSeek-V3` |
| `temperature` | float | 控制输出随机性 | 0.7（平衡） |
| `max_tokens` | int | 最大回复长度 | 2000 |
| `top_p` | float | 核采样参数 | 0.95 |

### Qwen配置

```yaml
llm:
  provider: "qwen"
  
  qwen:
    api_key: "sk-..."
    base_url: "https://dashscope.aliyuncs.com/api/v1"
    model: "qwen-turbo"
    temperature: 0.7
    max_tokens: 2000
    top_p: 0.8
```

### 系统提示词

```yaml
llm:
  system_prompt: |
    你是一个专业的物理老师。
    用简洁、准确的语言讲解概念。
    鼓励学生独立思考。
```

## 🔧 高级用法

### 自定义系统提示词

根据不同场景设置不同的系统提示词：

```python
# 数学老师
llm.set_system_prompt("你是一个专业的数学老师，用简洁清晰的语言解释概念。")

# 编程老师
llm.set_system_prompt("你是一个编程老师，帮助学生学习编程知识。")

# 通用助手
llm.set_system_prompt("你是一个友好的智能助手。")
```

### 调整对话参数

```python
# 在配置文件中修改
config['llm']['deepseek']['temperature'] = 0.9  # 提高创造性
config['llm']['deepseek']['max_tokens'] = 4000  # 增加回复长度

# 重新初始化
llm = LLMInterface(config['llm'])
```

### 管理长对话

```python
# 对话很多轮后，自动修剪历史
if len(llm.get_history()) > 20:
    llm.trim_history(max_turns=10)
```

## ❓ 常见问题

### Q1: API Key未配置

**现象**：初始化时警告 "API Key未配置"

**解决**：
```yaml
llm:
  deepseek:
    api_key: "sk-your-actual-key"  # 填入正确的API Key
```

### Q2: API调用失败

**现象**：错误信息包含 "401", "403" 或 "Invalid API Key"

**解决**：
1. 检查API Key是否正确
2. 检查账户余额（登录 https://siliconflow.cn/）
3. 检查网络连接
4. 查看详细日志：`cat data/logs/system.log`

### Q3: 回复长度不够

**解决**：
```yaml
llm:
  deepseek:
    max_tokens: 4000  # 增加最大回复长度
```

### Q4: 回复太随机或太确定

**解决**：
```yaml
# 提高创造性（更随机）
llm:
  deepseek:
    temperature: 0.9

# 降低创造性（更确定）
llm:
  deepseek:
    temperature: 0.3
```

### Q5: 如何切换提供商？

**解决**：
```python
# 方法1：在代码中切换
llm.switch_provider("qwen")

# 方法2：在配置文件中修改
# config.yaml:
#   provider: "qwen"
```

## 📊 性能优化

### 减少API调用

```python
# 批量处理多个问题（在历史中）
llm.chat("问题1")
llm.chat("问题2")
llm.chat("问题3")
# 只需要3次API调用
```

### 控制历史长度

```python
# 定期修剪历史，避免token过多
if len(llm.get_history()) > 10:
    llm.trim_history(max_turns=5)
```

## 🔗 相关资源

- **硅基流动**: https://siliconflow.cn/
- **DeepSeek**: https://www.deepseek.com/
- **Qwen**: https://github.com/QwenLM/Qwen
- **OpenAI API文档**: https://platform.openai.com/docs/

## 📝 注意事项

1. **API Key安全**：不要将包含API Key的配置提交到Git
2. **账户余额**：确保API账户有足够的余额
3. **历史管理**：长对话会消耗更多token，定期修剪历史
4. **网络连接**：确保可以访问API服务器
5. **错误处理**：API调用可能失败，建议添加重试逻辑

## 📄 许可证

MIT License

