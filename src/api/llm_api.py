"""
LLM API服务
提供大语言模型对话接口
"""

import os
import logging
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .models import (
    LLMChatRequest, LLMChatResponse,
    LLMHistoryResponse, LLMSystemPromptRequest,
    StatusResponse, ModelInfoResponse
)
from .utils import load_config, setup_logging
from ..llm.llm_interface import LLMInterface

setup_logging()
logger = logging.getLogger(__name__)

llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("LLM API服务启动中...")
    yield
    logger.info("LLM API服务关闭中...")
    global llm_client
    if llm_client:
        del llm_client


app = FastAPI(
    title="LLM API",
    description="大语言模型对话服务API - 支持DeepSeek/Qwen",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_model=StatusResponse)
def read_root():
    """根路径"""
    return StatusResponse(
        success=True,
        message="欢迎使用LLM对话"
    )


@app.post("/load", response_model=StatusResponse)
def load_model():
    """初始化LLM客户端"""
    global llm_client
    
    if llm_client is not None:
        return StatusResponse(
            success=True,
            message="LLM客户端已初始化，无需重复初始化"
        )
    
    try:
        logger.info("正在初始化LLM客户端...")

        config = load_config()
        llm_config = config.get('llm', {})

        llm_client = LLMInterface(llm_config)
        
        logger.info("LLM客户端初始化成功")
        return StatusResponse(
            success=True,
            message=f"LLM客户端初始化成功 (提供商: {llm_client.provider})"
        )
        
    except Exception as e:
        logger.error(f"初始化LLM客户端失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")


@app.post("/unload", response_model=StatusResponse)
def unload_model():
    """清理LLM客户端"""
    global llm_client
    
    if llm_client is None:
        return StatusResponse(
            success=True,
            message="LLM客户端未初始化"
        )
    
    try:
        llm_client.clear_history()
        del llm_client
        llm_client = None
        logger.info("LLM客户端已清理")
        
        return StatusResponse(
            success=True,
            message="LLM客户端已清理"
        )
        
    except Exception as e:
        logger.error(f"清理客户端失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """获取模型信息"""
    global llm_client
    
    if llm_client is None:
        return ModelInfoResponse(
            success=False,
            message="LLM客户端未初始化，请先调用 /load 接口"
        )
    
    try:
        info = {
            "provider": llm_client.provider,
            "model_name": llm_client.model_name,
            "temperature": llm_client.temperature,
            "max_tokens": llm_client.max_tokens,
            "top_p": llm_client.top_p,
            "system_prompt": llm_client.system_prompt,
            "history_length": len(llm_client.conversation_history)
        }
        return ModelInfoResponse(
            success=True,
            info=info
        )
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=LLMChatResponse)
def chat(request: LLMChatRequest):
    """
    对话接口
    
    支持带历史记录的多轮对话
    """
    global llm_client
    
    if llm_client is None:
        raise HTTPException(
            status_code=500,
            detail="LLM客户端未初始化，请先调用 /load 接口"
        )
    
    try:
        start_time = time.time()
        
        original_prompt = None
        if request.system_prompt:
            original_prompt = llm_client.system_prompt
            llm_client.set_system_prompt(request.system_prompt)
        
        if request.temperature is not None:
            original_temp = llm_client.temperature
            llm_client.temperature = request.temperature
        
        if request.max_tokens is not None:
            original_max_tokens = llm_client.max_tokens
            llm_client.max_tokens = request.max_tokens
  
        logger.info(f"用户消息: {request.message[:50]}...")
        reply = llm_client.chat(
            user_message=request.message,
            use_history=request.use_history
        )
        
        duration = time.time() - start_time
        logger.info(f"LLM回复完成，耗时: {duration:.2f}秒")

        if original_prompt:
            llm_client.set_system_prompt(original_prompt)
        
        if request.temperature is not None:
            llm_client.temperature = original_temp
        
        if request.max_tokens is not None:
            llm_client.max_tokens = original_max_tokens
        
        return LLMChatResponse(
            success=True,
            reply=reply,
            duration=duration
        )
        
    except Exception as e:
        logger.error(f"对话失败: {str(e)}")
        return LLMChatResponse(
            success=False,
            reply=None,
            message=str(e)
        )


@app.get("/history", response_model=LLMHistoryResponse)
def get_history():
    """获取对话历史"""
    global llm_client
    
    if llm_client is None:
        raise HTTPException(
            status_code=500,
            detail="LLM客户端未初始化"
        )
    
    try:
        history = llm_client.get_history()
        return LLMHistoryResponse(
            success=True,
            history=history,
            count=len(history)
        )
        
    except Exception as e:
        logger.error(f"获取历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history/clear", response_model=StatusResponse)
def clear_history():
    """清空对话历史"""
    global llm_client
    
    if llm_client is None:
        raise HTTPException(
            status_code=500,
            detail="LLM客户端未初始化"
        )
    
    try:
        llm_client.clear_history()
        return StatusResponse(
            success=True,
            message="对话历史已清空"
        )
        
    except Exception as e:
        logger.error(f"清空历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history/trim", response_model=StatusResponse)
def trim_history(max_turns: int = 10):
    """
    修剪对话历史
    
    Args:
        max_turns: 最大保留轮数
    """
    global llm_client
    
    if llm_client is None:
        raise HTTPException(
            status_code=500,
            detail="LLM客户端未初始化"
        )
    
    try:
        llm_client.trim_history(max_turns=max_turns)
        return StatusResponse(
            success=True,
            message=f"对话历史已修剪至{max_turns}轮"
        )
        
    except Exception as e:
        logger.error(f"修剪历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system_prompt", response_model=StatusResponse)
def set_system_prompt(request: LLMSystemPromptRequest):
    """设置系统提示词"""
    global llm_client
    
    if llm_client is None:
        raise HTTPException(
            status_code=500,
            detail="LLM客户端未初始化"
        )
    
    try:
        llm_client.set_system_prompt(request.system_prompt)
        return StatusResponse(
            success=True,
            message="系统提示词已更新"
        )
        
    except Exception as e:
        logger.error(f"设置系统提示词失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/switch_provider", response_model=StatusResponse)
def switch_provider(provider: str):
    """
    切换LLM提供商
    
    Args:
        provider: 提供商名称 (deepseek/qwen)
    """
    global llm_client
    
    if llm_client is None:
        raise HTTPException(
            status_code=500,
            detail="LLM客户端未初始化"
        )
    
    try:
        llm_client.switch_provider(provider)
        return StatusResponse(
            success=True,
            message=f"已切换到提供商: {provider}"
        )
        
    except Exception as e:
        logger.error(f"切换提供商失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "client_initialized": llm_client is not None
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("LLM_API_HOST", "0.0.0.0")
    port = int(os.getenv("LLM_API_PORT", "8002"))
    
    logger.info(f"启动LLM API服务: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
