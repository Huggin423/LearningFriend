"""
FunASR语音识别模块
使用阿里达摩院的FunASR进行中文语音识别
"""

import os
import logging
from typing import Optional, Union, Dict, Any, List
import numpy as np
import torch

# 先初始化logger，再在try-except中使用
logger = logging.getLogger(__name__)

try:
    from funasr import AutoModel
    HAS_FUNASR = True
except ImportError:
    HAS_FUNASR = False
    logger.warning("FunASR未安装，ASR功能将不可用")


class FunASRModule:
    """FunASR语音识别模块"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FunASR模块
        
        Args:
            config: ASR配置字典
        """
        if not HAS_FUNASR:
            raise ImportError(
                "FunASR未安装！请运行以下命令安装：\n"
                "  cd FunASR\n"
                "  pip install -e .\n"
                "或者：pip install funasr"
            )
        
        self.config = config
        self.model_name = config.get('model_name', 'paraformer-zh')
        self.model_revision = config.get('model_revision', 'v2.0.4')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('batch_size', 1)
        self.sample_rate = config.get('sample_rate', 16000)
        self.hotword = config.get('hotword', '')
        self.use_itn = config.get('use_itn', True)
        
        # VAD和Punctuation配置（可选）
        self.vad_model = config.get('vad_model', None)
        self.vad_kwargs = config.get('vad_kwargs', {})
        self.punc_model = config.get('punc_model', None)
        self.punc_kwargs = config.get('punc_kwargs', {})
        
        # Hub配置
        self.hub = config.get('hub', 'ms')  # ms=ModelScope, hf=HuggingFace
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载FunASR模型"""
        try:
            logger.info(f"正在加载FunASR模型: {self.model_name}")
            logger.info(f"使用设备: {self.device}")
            
            # 构建AutoModel参数
            model_kwargs = {
                'model': self.model_name,
                'model_revision': self.model_revision,
                'device': self.device,
                'batch_size': self.batch_size,
                'disable_update': True,  # 禁用模型自动更新
                'hub': self.hub,  # 模型仓库来源
            }
            
            # 可选添加VAD模型
            if self.vad_model:
                logger.info(f"启用VAD模型: {self.vad_model}")
                model_kwargs['vad_model'] = self.vad_model
                if self.vad_kwargs:
                    model_kwargs['vad_kwargs'] = self.vad_kwargs
            
            # 可选添加标点恢复模型
            if self.punc_model:
                logger.info(f"启用标点模型: {self.punc_model}")
                model_kwargs['punc_model'] = self.punc_model
                if self.punc_kwargs:
                    model_kwargs['punc_kwargs'] = self.punc_kwargs
            
            # 使用AutoModel加载模型
            self.model = AutoModel(**model_kwargs)
            
            logger.info("FunASR模型加载成功")
            
        except Exception as e:
            logger.error(f"加载FunASR模型失败: {str(e)}")
            # 提供更详细的错误信息
            if "Cannot find" in str(e) or "not found" in str(e).lower():
                logger.error(
                    "请确保模型名称正确，或检查网络连接。" +
                    "\n常用的模型名称："
                    "\n  - paraformer-zh: 中文语音识别"
                    "\n  - paraformer-zh-streaming: 实时中文识别"
                    "\n  - fsmn-vad: 语音活动检测"
                    "\n  - ct-punc: 标点恢复"
                )
            raise
    
    def transcribe(
        self, 
        audio_input: Union[str, np.ndarray, List[np.ndarray]],
        language: str = "zh",
        use_itn: Optional[bool] = None
    ) -> str:
        """
        语音转文字
        
        Args:
            audio_input: 音频文件路径、音频数组或数组列表
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
            
            logger.debug(f"开始识别音频: {audio_input if isinstance(audio_input, str) else f'numpy array (shape: {np.array(audio_input).shape if isinstance(audio_input, (np.ndarray, list)) else "unknown"})'}")
            
            # 调用FunASR模型进行识别
            # generate方法的参数根据FunASR实际API调整
            result = self.model.generate(
                input=audio_input,
                batch_size=self.batch_size,
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
    
    def transcribe_batch(
        self,
        audio_inputs: List[Union[str, np.ndarray]],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        批量识别音频
        
        Args:
            audio_inputs: 音频文件路径或数组列表
            batch_size: 批处理大小，None则使用配置值
        
        Returns:
            识别文本列表
        """
        if not isinstance(audio_inputs, list):
            audio_inputs = [audio_inputs]
        
        results = []
        batch_sz = batch_size if batch_size is not None else self.batch_size
        
        # 分批处理
        for i in range(0, len(audio_inputs), batch_sz):
            batch = audio_inputs[i:i+batch_sz]
            logger.debug(f"处理批次 {i//batch_sz + 1}/{(len(audio_inputs)-1)//batch_sz + 1}")
            
            for audio in batch:
                text = self.transcribe(audio)
                results.append(text)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'model_name': self.model_name,
            'model_revision': self.model_revision,
            'device': self.device,
            'sample_rate': self.sample_rate,
            'has_vad': self.vad_model is not None,
            'has_punctuation': self.punc_model is not None,
            'hub': self.hub
        }
        return info
    
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

