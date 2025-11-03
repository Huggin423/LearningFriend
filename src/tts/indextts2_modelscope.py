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
        """从 ModelScope 下载模型，然后使用 IndexTTS2 官方代码加载"""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            
            logger.info("正在从 ModelScope 下载 IndexTTS2 模型...")
            
            # 下载或加载模型
            model_id = "IndexTeam/IndexTTS-2"
            
            # 检查模型是否已下载
            if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
                logger.info("模型未找到，正在从 ModelScope 下载...")
                try:
                    snapshot_download(
                        model_id, 
                        cache_dir=str(self.model_dir),
                        local_files_only=False
                    )
                    logger.info("✓ 模型下载完成")
                except Exception as e:
                    logger.warning(f"下载到指定目录失败: {e}")
                    # 检查是否在默认缓存位置
                    import os
                    default_cache = os.path.expanduser("~/.cache/modelscope/hub/" + model_id.replace("/", "--"))
                    if os.path.exists(default_cache):
                        logger.info(f"使用 ModelScope 默认缓存位置: {default_cache}")
                        self.model_dir = Path(default_cache)
            
            # 确保模型目录存在且有文件
            if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
                raise FileNotFoundError(
                    f"模型目录为空或不存在: {self.model_dir}\n"
                    f"请手动下载模型或检查 ModelScope 配置"
                )
            
            # 查找配置文件
            config_path = None
            possible_config_paths = [
                self.model_dir / "config.yaml",
                self.model_dir / "IndexTeam" / "IndexTTS-2" / "config.yaml",
            ]
            
            for path in possible_config_paths:
                if path.exists():
                    config_path = path
                    logger.info(f"找到配置文件: {config_path}")
                    break
            
            if config_path is None:
                # 如果找不到配置文件，尝试在模型目录的子目录中查找
                for subdir in self.model_dir.iterdir():
                    if subdir.is_dir():
                        sub_config = subdir / "config.yaml"
                        if sub_config.exists():
                            config_path = sub_config
                            self.model_dir = subdir  # 更新模型目录
                            logger.info(f"在子目录中找到配置文件: {config_path}")
                            break
            
            if config_path is None:
                raise FileNotFoundError(
                    f"找不到配置文件 config.yaml\n"
                    f"请检查模型目录: {self.model_dir}"
                )
            
            # 添加 index-tts 到 Python 路径
            project_root = Path(__file__).parent.parent.parent
            indextts_path = project_root / "index-tts"
            if indextts_path.exists():
                if str(indextts_path) not in sys.path:
                    sys.path.insert(0, str(indextts_path))
            else:
                logger.warning(f"index-tts 目录不存在: {indextts_path}")
                logger.warning("尝试使用已安装的 indextts 包")
            
            # 使用 IndexTTS2 官方代码加载模型
            logger.info("使用 IndexTTS2 官方代码加载模型...")
            from indextts.infer_v2 import IndexTTS2
            
            # 初始化 IndexTTS2
            self.tts_model = IndexTTS2(
                cfg_path=str(config_path),
                model_dir=str(self.model_dir),
                use_fp16=self.config.get('use_fp16', False),
                device=self.device,
                use_cuda_kernel=self.config.get('use_cuda_kernel', None),
            )
            
            logger.info("✓ 模型加载成功")
            
        except ImportError as e:
            logger.error(f"导入失败: {str(e)}")
            logger.error("请确保已安装所需依赖:")
            logger.error("  pip install modelscope")
            logger.error("  pip install indextts (如果可用)")
            raise
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
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
            speed: 语速倍率（注意：IndexTTS2 可能不支持动态语速调整）
            **kwargs: 其他参数
        
        Returns:
            audio: 音频数组
        """
        try:
            import tempfile
            import soundfile as sf
            import torch
            
            # 准备参考音频路径
            temp_file = None
            if reference_audio_path is None and reference_audio is not None:
                # 保存临时文件
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, reference_audio, self.sample_rate)
                reference_audio_path = temp_file.name
            
            # 如果没有提供参考音频，尝试使用默认参考音频
            if reference_audio_path is None:
                # 检查是否有默认参考音频（可以从配置中读取）
                default_ref_audio = self.config.get('default_reference_audio')
                if default_ref_audio and os.path.exists(default_ref_audio):
                    reference_audio_path = default_ref_audio
                    logger.info(f"使用默认参考音频: {reference_audio_path}")
                else:
                    # 如果完全没有参考音频，IndexTTS2 可能无法工作
                    # 生成一个简单的提示音作为参考（不推荐，但可以作为后备方案）
                    logger.warning("未提供参考音频，IndexTTS2 需要参考音频才能工作")
                    logger.warning("请提供 reference_audio 或 reference_audio_path")
                    raise ValueError("IndexTTS2 需要参考音频才能进行语音合成")
            
            # 创建临时输出文件
            output_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = output_temp.name
            output_temp.close()
            
            try:
                # 调用 IndexTTS2 的 infer 方法
                # infer 方法需要: spk_audio_prompt, text, output_path
                result = self.tts_model.infer(
                    spk_audio_prompt=reference_audio_path,
                    text=text,
                    output_path=output_path,
                    emo_audio_prompt=None,  # 可以后续支持
                    emo_alpha=emotion_strength if emotion else 1.0,
                    emo_vector=None,  # 可以后续支持
                    verbose=False,
                    **kwargs
                )
                
                # 读取生成的音频文件
                if os.path.exists(output_path):
                    audio, sr = sf.read(output_path)
                    # 确保采样率匹配
                    if sr != self.sample_rate:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    
                    # 转换为 float32
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    
                    logger.info(f"合成成功，音频长度: {len(audio)/self.sample_rate:.2f}秒")
                    return audio
                else:
                    raise FileNotFoundError(f"生成的音频文件不存在: {output_path}")
                    
            finally:
                # 清理临时文件
                if temp_file is not None and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"语音合成失败: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
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
