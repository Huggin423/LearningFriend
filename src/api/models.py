"""
API数据模型定义
使用Pydantic进行数据验证
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ==================== ASR 相关模型 ====================

class ASRRequest(BaseModel):
    """ASR识别请求"""
    audio_path: Optional[str] = Field(None, description="音频文件路径（本地文件）")
    audio_base64: Optional[str] = Field(None, description="Base64编码的音频数据")
    language: str = Field("zh", description="语言代码")
    use_itn: bool = Field(True, description="是否使用逆文本归一化")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_path": "/path/to/audio.wav",
                "language": "zh",
                "use_itn": True
            }
        }


class ASRBatchRequest(BaseModel):
    """ASR批量识别请求"""
    audio_paths: List[str] = Field(..., description="音频文件路径列表")
    language: str = Field("zh", description="语言代码")
    use_itn: bool = Field(True, description="是否使用逆文本归一化")
    batch_size: Optional[int] = Field(None, description="批处理大小")


class ASRResponse(BaseModel):
    """ASR识别响应"""
    success: bool = Field(..., description="是否成功")
    text: Optional[str] = Field(None, description="识别出的文本")
    message: Optional[str] = Field(None, description="错误或提示信息")
    duration: Optional[float] = Field(None, description="处理耗时（秒）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "text": "这是识别出的文本内容",
                "message": None,
                "duration": 0.85
            }
        }


class ASRBatchResponse(BaseModel):
    """ASR批量识别响应"""
    success: bool = Field(..., description="是否成功")
    results: Optional[List[str]] = Field(None, description="识别文本列表")
    message: Optional[str] = Field(None, description="错误或提示信息")
    duration: Optional[float] = Field(None, description="总处理耗时（秒）")


# ==================== LLM 相关模型 ====================

class LLMChatRequest(BaseModel):
    """LLM对话请求"""
    message: str = Field(..., description="用户消息", min_length=1)
    use_history: bool = Field(True, description="是否使用对话历史")
    system_prompt: Optional[str] = Field(None, description="临时系统提示词（覆盖默认）")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(None, gt=0, description="最大token数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "你好，请介绍一下自己",
                "use_history": True,
                "temperature": 0.7
            }
        }


class LLMChatResponse(BaseModel):
    """LLM对话响应"""
    success: bool = Field(..., description="是否成功")
    reply: Optional[str] = Field(None, description="LLM回复")
    message: Optional[str] = Field(None, description="错误或提示信息")
    duration: Optional[float] = Field(None, description="处理耗时（秒）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "reply": "你好！我是一个AI助手...",
                "message": None,
                "duration": 1.23
            }
        }


class LLMHistoryResponse(BaseModel):
    """对话历史响应"""
    success: bool = Field(..., description="是否成功")
    history: Optional[List[Dict[str, str]]] = Field(None, description="对话历史")
    count: Optional[int] = Field(None, description="消息数量")


class LLMSystemPromptRequest(BaseModel):
    """设置系统提示词请求"""
    system_prompt: str = Field(..., description="新的系统提示词", min_length=1)


# ==================== 通用模型 ====================

class StatusResponse(BaseModel):
    """状态响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="状态信息")


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    success: bool = Field(..., description="是否成功")
    info: Optional[Dict[str, Any]] = Field(None, description="模型信息")
    message: Optional[str] = Field(None, description="错误或提示信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    services: Dict[str, bool] = Field(..., description="各服务状态")
    version: str = Field(..., description="API版本")
