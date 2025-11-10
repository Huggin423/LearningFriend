"""
主API服务
整合ASR、LLM、TTS服务，提供统一入口
"""

import os
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .models import (
    ASRRequest, ASRResponse,
    LLMChatRequest, LLMChatResponse,
    StatusResponse, HealthResponse
)
from .utils import (
    decode_base64_audio, cleanup_temp_file,
    validate_audio_file, load_config, setup_logging
)
from ..asr.funasr_module import FunASRModule
from ..llm.llm_interface import LLMInterface

setup_logging()
logger = logging.getLogger(__name__)

asr_model = None
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("主API服务启动中...")
    yield
    logger.info("主API服务关闭中...")
    global asr_model, llm_client
    if asr_model:
        del asr_model
    if llm_client:
        del llm_client

app = FastAPI(
    title="LearningFriend API",
    description="学习伙伴项目统一API - ASR + LLM + TTS",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=StatusResponse)
def read_root():
    """根路径"""
    return StatusResponse(
        success=True,
        message="欢迎使用LearningFriend API - 语音识别、对话、语音合成统一服务"
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        services={
            "asr": asr_model is not None,
            "llm": llm_client is not None,
        },
        version="1.0.0"
    )


# ==================== 加载/卸载模型 ====================

@app.post("/load/asr", response_model=StatusResponse)
def load_asr():
    """加载ASR模型"""
    global asr_model
    
    if asr_model is not None:
        return StatusResponse(
            success=True,
            message="ASR模型已加载"
        )
    
    try:
        logger.info("正在加载ASR模型...")
        config = load_config()
        asr_config = config.get('asr', {})
        asr_model = FunASRModule(asr_config)
        
        logger.info("ASR模型加载成功")
        return StatusResponse(
            success=True,
            message="ASR模型加载成功"
        )
        
    except Exception as e:
        logger.error(f"加载ASR模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载失败: {str(e)}")


@app.post("/load/llm", response_model=StatusResponse)
def load_llm():
    """初始化LLM客户端"""
    global llm_client
    
    if llm_client is not None:
        return StatusResponse(
            success=True,
            message="LLM客户端已初始化"
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


@app.post("/load/all", response_model=StatusResponse)
def load_all():
    """加载所有模型"""
    try:
        load_asr()
        load_llm()
        
        return StatusResponse(
            success=True,
            message="所有服务已加载"
        )
        
    except Exception as e:
        logger.error(f"加载服务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载失败: {str(e)}")


@app.post("/unload/asr", response_model=StatusResponse)
def unload_asr():
    """卸载ASR模型"""
    global asr_model
    
    if asr_model is None:
        return StatusResponse(success=True, message="ASR模型未加载")
    
    try:
        del asr_model
        asr_model = None
        return StatusResponse(success=True, message="ASR模型已卸载")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload/llm", response_model=StatusResponse)
def unload_llm():
    """清理LLM客户端"""
    global llm_client
    
    if llm_client is None:
        return StatusResponse(success=True, message="LLM客户端未初始化")
    
    try:
        llm_client.clear_history()
        del llm_client
        llm_client = None
        return StatusResponse(success=True, message="LLM客户端已清理")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ASR接口 ====================

@app.post("/asr/transcribe", response_model=ASRResponse)
def asr_transcribe(request: ASRRequest):
    """
    语音识别接口
    
    可以通过 audio_path 或 audio_base64 提供音频
    """
    global asr_model
    
    if asr_model is None:
        raise HTTPException(
            status_code=500,
            detail="ASR模型未加载，请先调用 /load/asr 接口"
        )
    
    temp_file = None
    
    try:
        start_time = time.time()
        
        if request.audio_path:
            audio_path = request.audio_path
            if not validate_audio_file(audio_path):
                raise HTTPException(
                    status_code=400,
                    detail="音频文件不存在或格式不支持"
                )
        elif request.audio_base64:
            temp_file = decode_base64_audio(request.audio_base64)
            audio_path = temp_file
        else:
            raise HTTPException(
                status_code=400,
                detail="请提供 audio_path 或 audio_base64"
            )
        
        logger.info(f"开始识别音频: {audio_path}")
        text = asr_model.transcribe(
            audio_input=audio_path,
            language=request.language,
            use_itn=request.use_itn
        )
        
        duration = time.time() - start_time
        logger.info(f"识别完成，耗时: {duration:.2f}秒")
        
        return ASRResponse(
            success=True,
            text=text,
            duration=duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"识别失败: {str(e)}")
        return ASRResponse(
            success=False,
            text=None,
            message=str(e)
        )
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)


# ==================== LLM接口 ====================

@app.post("/llm/chat", response_model=LLMChatResponse)
def llm_chat(request: LLMChatRequest):
    """
    对话接口
    
    支持带历史记录的多轮对话
    """
    global llm_client
    
    if llm_client is None:
        raise HTTPException(
            status_code=500,
            detail="LLM客户端未初始化，请先调用 /load/llm 接口"
        )
    
    try:
        start_time = time.time()
        
        original_prompt = None
        if request.system_prompt:
            original_prompt = llm_client.system_prompt
            llm_client.set_system_prompt(request.system_prompt)
        
        logger.info(f"用户消息: {request.message[:50]}...")
        reply = llm_client.chat(
            user_message=request.message,
            use_history=request.use_history
        )
        
        duration = time.time() - start_time
        logger.info(f"LLM回复完成，耗时: {duration:.2f}秒")

        if original_prompt:
            llm_client.set_system_prompt(original_prompt)
        
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


@app.post("/llm/history/clear", response_model=StatusResponse)
def llm_clear_history():
    """清空对话历史"""
    global llm_client
    
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM客户端未初始化")
    
    try:
        llm_client.clear_history()
        return StatusResponse(success=True, message="对话历史已清空")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 综合接口 ====================

@app.post("/conversation", response_model=LLMChatResponse)
def conversation(request: ASRRequest):
    """
    对话流程接口：语音 -> 文字 -> LLM -> 回复文字
    
    输入音频，返回LLM的文字回复
    """
    global asr_model, llm_client
    
    if asr_model is None:
        raise HTTPException(status_code=500, detail="ASR模型未加载")
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM客户端未初始化")
    
    temp_file = None
    
    try:
        start_time = time.time()
 
        if request.audio_path:
            audio_path = request.audio_path
            if not validate_audio_file(audio_path):
                raise HTTPException(status_code=400, detail="音频文件无效")
        elif request.audio_base64:
            temp_file = decode_base64_audio(request.audio_base64)
            audio_path = temp_file
        else:
            raise HTTPException(status_code=400, detail="请提供音频")
        
        logger.info("步骤1: 语音识别...")
        user_text = asr_model.transcribe(
            audio_input=audio_path,
            language=request.language,
            use_itn=request.use_itn
        )
        logger.info(f"识别结果: {user_text}")
        
        logger.info("步骤2: LLM对话...")
        reply = llm_client.chat(
            user_message=user_text,
            use_history=True
        )
        
        duration = time.time() - start_time
        logger.info(f"对话流程完成，总耗时: {duration:.2f}秒")
        
        return LLMChatResponse(
            success=True,
            reply=reply,
            message=f"用户说: {user_text}",
            duration=duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话流程失败: {str(e)}")
        return LLMChatResponse(
            success=False,
            reply=None,
            message=str(e)
        )
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"启动主API服务: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
