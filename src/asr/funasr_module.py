"""
FunASR语音识别模块
使用阿里达摩院的FunASR进行中文语音识别
"""

import os
import logging
from typing import Optional, Union, Dict, Any
import numpy as np
from funasr import AutoModel


logger = logging.getLogger(__name__)


class FunASRModule:
    """FunASR语音识别模块"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FunASR模块
        
        Args:
            config: ASR配置字典
        """
        self.config = config
        self.model_name = config.get('model_name', 'paraformer-zh')
        self.model_revision = config.get('model_revision', 'v2.0.4')
        self.device = config.get('device', 'cuda')
        self.batch_size = config.get('batch_size', 1)
        self.sample_rate = config.get('sample_rate', 16000)
        self.hotword = config.get('hotword', '')
        self.use_itn = config.get('use_itn', True)
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载FunASR模型"""
        try:
            logger.info(f"正在加载FunASR模型: {self.model_name}")
            
            # 使用AutoModel加载模型
            self.model = AutoModel(
                model=self.model_name,
                model_revision=self.model_revision,
                device=self.device,
                batch_size=self.batch_size,
                disable_update=True,  # 禁用模型自动更新
            )
            
            logger.info("FunASR模型加载成功")
            
        except Exception as e:
            logger.error(f"加载FunASR模型失败: {str(e)}")
            raise
    
    def transcribe(
        self, 
        audio_input: Union[str, np.ndarray],
        language: str = "zh",
        use_itn: Optional[bool] = None
    ) -> str:
        """
        语音转文字
        
        Args:
            audio_input: 音频文件路径或音频数组
            language: 语言代码，默认为中文
            use_itn: 是否使用逆文本归一化，None则使用配置文件设置
            
        Returns:
            识别出的文本
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用_load_model()")
        
        try:
            # 如果use_itn未指定，使用配置文件设置
            if use_itn is None:
                use_itn = self.use_itn
            
            logger.debug(f"开始识别音频: {audio_input if isinstance(audio_input, str) else 'numpy array'}")
            
            # 调用FunASR模型进行识别
            result = self.model.generate(
                input=audio_input,
                batch_size=self.batch_size,
                hotword=self.hotword,
            )
            
            # 提取识别文本
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('text', '')
            elif isinstance(result, dict):
                text = result.get('text', '')
            else:
                text = str(result)
            
            logger.info(f"识别结果: {text}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"语音识别失败: {str(e)}")
            raise
    
    def transcribe_file(self, audio_path: str) -> str:
        """
        从音频文件识别文本
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            识别出的文本
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        return self.transcribe(audio_path)
    
    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = None) -> str:
        """
        从音频数组识别文本
        
        Args:
            audio_array: 音频数据数组
            sample_rate: 采样率，如果与配置不同需要指定
            
        Returns:
            识别出的文本
        """
        if sample_rate and sample_rate != self.sample_rate:
            logger.warning(f"音频采样率({sample_rate})与配置({self.sample_rate})不一致")
        
        return self.transcribe(audio_array)
    
    def set_hotword(self, hotword: str):
        """
        设置热词
        
        Args:
            hotword: 热词字符串，用空格分隔多个热词
        """
        self.hotword = hotword
        logger.info(f"设置热词: {hotword}")
    
    def __del__(self):
        """析构函数，清理资源"""
        if self.model is not None:
            del self.model
            logger.debug("FunASR模型已卸载")

