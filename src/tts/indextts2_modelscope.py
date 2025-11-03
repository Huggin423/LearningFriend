"""
IndexTTS2 ModelScope 官方模型包装器
直接使用 ModelScope 上的预训练模型
"""

import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class IndexTTS2ModelScope:
    """IndexTTS2 ModelScope 官方模型包装器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化官方IndexTTS2模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = config.get('sample_rate', 22050)
        self.speed = config.get('speed', 1.0)
        
        # 模型路径
        self.model_dir = Path(config.get('model_path', 'models/indextts2'))
        
        # 初始化模型
        self._load_model()
        
        logger.info("IndexTTS2 ModelScope模型初始化成功")
        logger.info(f"设备: {self.device}")
        logger.info(f"模型目录: {self.model_dir}")
    
    def _load_model(self):
        """从 ModelScope 加载模型"""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            logger.info("正在从 ModelScope 加载 IndexTTS2 模型...")
            
            # 下载或加载模型
            model_id = "IndexTeam/IndexTTS-2"
            
            # 检查模型是否已下载
            if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
                logger.info("模型未找到，正在从 ModelScope 下载...")
                snapshot_download(
                    model_id, 
                    cache_dir=str(self.model_dir),
                    local_files_only=False
                )
            
            # 创建 TTS pipeline
            self.pipeline = pipeline(
                Tasks.text_to_speech,
                model=str(self.model_dir),
                device=self.device
            )
            
            logger.info("✓ 模型加载成功")
            
        except ImportError:
            logger.error("ModelScope 未安装，请运行: pip install modelscope")
            raise
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def synthesize(
        self,
        text: str,
        reference_audio: Optional[np.ndarray] = None,
        reference_audio_path: Optional[str] = None,
        emotion: Optional[str] = None,
        emotion_strength: float = 1.0,
        speed: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        语音合成
        
        Args:
            text: 要合成的文本
            reference_audio: 参考音频数组 (可选)
            reference_audio_path: 参考音频文件路径 (可选)
            emotion: 情感标签 (可选): 'neutral', 'happiness', 'sadness', 'anger', etc.
            emotion_strength: 情感强度 (0-1)
            speed: 语速倍率
            **kwargs: 其他参数
        
        Returns:
            audio: 音频数组
        """
        try:
            if speed is None:
                speed = self.speed
            
            # 准备输入数据
            input_data = {
                'text': text
            }
            
            # 添加参考音频（如果提供）
            if reference_audio_path is None and reference_audio is not None:
                # 保存临时文件
                import tempfile
                import soundfile as sf
                
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, reference_audio, self.sample_rate)
                reference_audio_path = temp_file.name
            
            if reference_audio_path:
                input_data['reference_audio'] = reference_audio_path
            
            # 添加情感参数
            if emotion:
                input_data['emotion'] = emotion
                input_data['emotion_strength'] = emotion_strength
            
            # 添加语速
            if speed and speed != 1.0:
                input_data['speed'] = speed
            
            # 调用 ModelScope pipeline
            try:
                output = self.pipeline(input_data)
                
                # 提取音频数据
                if isinstance(output, dict):
                    audio = output.get('output_wav', output.get('audio'))
                else:
                    audio = output
                
                # 清理临时文件
                if 'temp_file' in locals():
                    os.unlink(temp_file.name)
                
                # 转换音频格式
                if isinstance(audio, str):
                    import soundfile as sf
                    audio, _ = sf.read(audio)
                elif not isinstance(audio, np.ndarray):
                    # 尝试转换为 numpy 数组
                    audio = np.array(audio)
                
                # 转换为 float32
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                
                logger.info(f"合成成功，音频长度: {len(audio)/self.sample_rate:.2f}秒")
                return audio
                
            except Exception as e:
                logger.error(f"模型推理失败: {str(e)}")
                # 清理临时文件
                if 'temp_file' in locals():
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                raise
            
        except Exception as e:
            logger.error(f"语音合成失败: {str(e)}")
            raise
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        **kwargs
    ) -> str:
        """
        合成语音并保存到文件
        
        Args:
            text: 要合成的文本
            output_path: 输出文件路径
            **kwargs: 其他参数
        
        Returns:
            output_path: 输出文件路径
        """
        import soundfile as sf
        
        # 生成音频
        audio = self.synthesize(text, **kwargs)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        
        logger.info(f"音频已保存到: {output_path}")
        return output_path
    
    def clone_voice(
        self,
        reference_audio: np.ndarray,
        text: str,
        **kwargs
    ) -> np.ndarray:
        """
        零样本语音克隆
        
        Args:
            reference_audio: 参考音频
            text: 要合成的文本
            **kwargs: 其他参数
        
        Returns:
            audio: 合成的音频
        """
        return self.synthesize(
            text=text,
            reference_audio=reference_audio,
            **kwargs
        )
    
    def set_speed(self, speed: float):
        """设置语速"""
        self.speed = speed
        logger.info(f"设置语速: {speed}")


# 改进的官方模型包装器：完整的接口兼容
class IndexTTS2ModelScopeWrapper(IndexTTS2ModelScope):
    """
    改进的官方模型包装器
    提供与复现模型完全兼容的接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 默认参数
        self.speaker_id = config.get('speaker_id', 0)
        self.emotion = config.get('emotion', 'neutral')
        self.pitch = config.get('pitch', 1.0)
    
    def synthesize(
        self,
        text: str,
        timbre_prompt: Optional[np.ndarray] = None,
        style_prompt: Optional[np.ndarray] = None,
        emotion: Optional[str] = None,
        target_duration: Optional[float] = None,
        speed: Optional[float] = None,
        speaker_id: Optional[int] = None,
        reference_audio: Optional[np.ndarray] = None,
        reference_audio_path: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        语音合成（兼容复现模型接口）
        
        Args:
            text: 要合成的文本
            timbre_prompt: 音色参考音频（兼容参数）
            style_prompt: 风格参考音频（兼容参数）
            emotion: 情感标签
            target_duration: 目标时长（官方模型可能不支持）
            speed: 语速倍率
            speaker_id: 说话人ID（兼容参数）
            reference_audio: 参考音频（官方接口）
            reference_audio_path: 参考音频路径（官方接口）
            **kwargs: 其他参数
        
        Returns:
            audio: 音频数组
        """
        # 统一参考音频处理
        ref_audio = reference_audio or timbre_prompt
        ref_path = reference_audio_path
        
        # 如果没有参考音频，尝试使用style_prompt
        if ref_audio is None and style_prompt is not None:
            ref_audio = style_prompt
        
        # 使用默认情感
        if emotion is None:
            emotion = self.emotion
        
        # 调用父类方法
        return super().synthesize(
            text=text,
            reference_audio=ref_audio,
            reference_audio_path=ref_path,
            emotion=emotion,
            speed=speed or self.speed,
            **kwargs
        )
    
    def synthesize_batch(
        self,
        texts: list,
        **kwargs
    ) -> list:
        """
        批量合成（兼容接口）
        
        Args:
            texts: 文本列表
            **kwargs: 其他参数
        
        Returns:
            audios: 音频数组列表
        """
        audios = []
        for i, text in enumerate(texts):
            logger.info(f"批量合成进度: {i+1}/{len(texts)}")
            audio = self.synthesize(text, **kwargs)
            audios.append(audio)
        return audios
    
    def set_speaker(self, speaker_id: int):
        """设置说话人（兼容接口）"""
        self.speaker_id = speaker_id
        logger.info(f"设置说话人ID: {speaker_id}（官方模型通过参考音频控制）")
    
    def set_emotion(self, emotion: str):
        """设置默认情感"""
        self.emotion = emotion
        logger.info(f"设置默认情感: {emotion}")
    
    def set_pitch(self, pitch: float):
        """设置音高（兼容接口，官方模型可能不支持）"""
        self.pitch = pitch
        logger.info(f"设置音高: {pitch}（注意：官方模型可能不支持此参数）")


# 兼容性别名
IndexTTSModule = IndexTTS2ModelScopeWrapper
