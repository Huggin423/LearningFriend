"""
ASR API服务
提供语音识别接口
"""

import os
import logging
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .models import (
    ASRRequest, ASRResponse,
    ASRBatchRequest, ASRBatchResponse,
    StatusResponse, ModelInfoResponse
)
from .utils import (
    decode_base64_audio, cleanup_temp_file,
    validate_audio_file, load_config, setup_logging
)
from ..asr.funasr_module import FunASRModule

setup_logging()
logger = logging.getLogger(__name__)

asr_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("ASR API服务启动中...")
    yield
    logger.info("ASR API服务关闭中...")
    global asr_model
    if asr_model:
        del asr_model


app = FastAPI(
    title="ASR API",
    description="语音识别服务API - 基于FunASR",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_model=StatusResponse)
def read_root():
    """根路径"""
    return StatusResponse(
        success=True,
        message="欢迎使用ASR语音识别"
    )


@app.post("/load", response_model=StatusResponse)
def load_model():
    """加载ASR模型"""
    global asr_model
    
    if asr_model is not None:
        return StatusResponse(
            success=True,
            message="模型已加载，无需重复加载"
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
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@app.post("/unload", response_model=StatusResponse)
def unload_model():
    """卸载ASR模型"""
    global asr_model
    if asr_model is None:
        return StatusResponse(
            success=True,
            message="模型未加载"
        )
    
    try:
        del asr_model
        asr_model = None
        logger.info("ASR模型已卸载")
        
        return StatusResponse(
            success=True,
            message="ASR模型已卸载"
        )
        
    except Exception as e:
        logger.error(f"卸载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")


@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """获取模型信息"""
    global asr_model
    
    if asr_model is None:
        return ModelInfoResponse(
            success=False,
            message="模型未加载，请先调用 /load 接口"
        )
    
    try:
        info = asr_model.get_model_info()
        return ModelInfoResponse(
            success=True,
            info=info
        )
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=ASRResponse)
def transcribe(request: ASRRequest):
    """
    语音识别接口
    
    可以通过 audio_path 或 audio_base64 提供音频
    """
    global asr_model
    
    if asr_model is None:
        raise HTTPException(
            status_code=500,
            detail="模型未加载，请先调用 /load 接口"
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


@app.post("/transcribe/batch", response_model=ASRBatchResponse)
def transcribe_batch(request: ASRBatchRequest):
    """批量语音识别接口"""
    global asr_model
    
    if asr_model is None:
        raise HTTPException(
            status_code=500,
            detail="模型未加载，请先调用 /load 接口"
        )
    
    try:
        start_time = time.time()
        
        for audio_path in request.audio_paths:
            if not validate_audio_file(audio_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"音频文件不存在或格式不支持: {audio_path}"
                )
        
        logger.info(f"开始批量识别，共 {len(request.audio_paths)} 个文件")
        results = asr_model.transcribe_batch(
            audio_inputs=request.audio_paths,
            batch_size=request.batch_size
        )
        
        duration = time.time() - start_time
        logger.info(f"批量识别完成，耗时: {duration:.2f}秒")
        
        return ASRBatchResponse(
            success=True,
            results=results,
            duration=duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量识别失败: {str(e)}")
        return ASRBatchResponse(
            success=False,
            results=None,
            message=str(e)
        )


@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": asr_model is not None
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ASR_API_HOST", "0.0.0.0")
    port = int(os.getenv("ASR_API_PORT", "8001"))
    
    logger.info(f"启动ASR API服务: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
