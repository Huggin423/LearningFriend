"""
IndexTTS2 官方模型集成包装器
直接使用 Hugging Face 上的预训练模型
"""

import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class IndexTTS2Official:
    """IndexTTS2官方模型包装器"""
    
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
        self.model_dir = Path(config.get('model_path', 'checkpoints'))
        
        # 初始化官方代码路径
        self.official_repo_path = Path(config.get('official_repo', 'index-tts'))
        
        # 检查并下载模型
        self._setup_official_model()
        
        # 加载官方推理接口
        self._load_official_inference()
        
        logger.info("IndexTTS2官方模型初始化成功")
        logger.info(f"设备: {self.device}")
        logger.info(f"模型目录: {self.model_dir}")
    
    def _setup_official_model(self):
        """设置官方模型"""
        if not self.official_repo_path.exists():
            logger.info("官方代码仓库不存在，正在克隆...")
            self._clone_official_repo()
        
        # 添加到Python路径
        if str(self.official_repo_path) not in sys.path:
            sys.path.insert(0, str(self.official_repo_path))
        
        # 检查模型文件
        if not self._check_model_files():
            logger.info("模型文件不完整，正在下载...")
            self._download_models()
    
    def _clone_official_repo(self):
        """克隆官方仓库"""
        import subprocess
        
        try:
            logger.info("克隆 IndexTTS 官方仓库...")
            subprocess.run([
                'git', 'clone',
                'https://github.com/index-tts/index-tts.git',
                str(self.official_repo_path)
            ], check=True)
            logger.info("✓ 仓库克隆成功")
            
            # 安装依赖（如果 requirements.txt 存在）
            requirements_file = self.official_repo_path / 'requirements.txt'
            if requirements_file.exists():
                logger.info("安装依赖...")
                try:
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install',
                        '-r', str(requirements_file)
                    ], check=True)
                    logger.info("✓ 依赖安装成功")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"依赖安装失败: {str(e)}，将继续尝试加载模型")
            else:
                logger.warning(f"未找到 requirements.txt 文件: {requirements_file}")
                logger.info("跳过依赖安装，将尝试直接使用官方代码")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"克隆仓库失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"设置官方仓库时发生错误: {str(e)}")
            raise
    
    def _check_model_files(self) -> bool:
        """检查模型文件是否完整"""
        required_files = [
            'config.yaml',
            'bpe.model',
            'feat1.pt',
            'feat2.pt',
            'qwen0.6bemo4-merge/model-00001-of-00002.safetensors',
            'qwen0.6bemo4-merge/model-00002-of-00002.safetensors',
        ]
        
        # 检查可能的路径（ModelScope 可能下载到子目录）
        possible_paths = [
            self.model_dir,
            self.model_dir / "IndexTeam" / "IndexTTS-2",
        ]
        
        # 找到实际的文件目录
        actual_dir = None
        for path in possible_paths:
            if (path / 'config.yaml').exists():
                actual_dir = path
                break
        
        if actual_dir is None:
            # 如果没有找到，使用默认路径
            actual_dir = self.model_dir
        
        # 检查文件是否存在
        for file in required_files:
            file_path = actual_dir / file
            if not file_path.exists():
                logger.warning(f"缺少文件: {file} (在 {actual_dir.relative_to(self.model_dir) if actual_dir != self.model_dir else 'checkpoints'} 目录)")
                return False
        
        # 如果文件在子目录中，更新 model_dir
        if actual_dir != self.model_dir:
            logger.info(f"检测到模型文件在子目录: {actual_dir.relative_to(self.model_dir)}")
            self.model_dir = actual_dir
        
        return True
    
    def _download_models(self):
        """从Hugging Face下载模型"""
        try:
            logger.info("从 Hugging Face 下载模型...")
            
            # 使用 huggingface-hub 下载
            try:
                from huggingface_hub import snapshot_download
                
                snapshot_download(
                    repo_id="IndexTeam/IndexTTS-2",
                    local_dir=str(self.model_dir),
                    local_dir_use_symlinks=False
                )
                logger.info("✓ 模型下载成功")
                
            except ImportError:
                logger.error("huggingface-hub 未安装，请运行: pip install huggingface-hub")
                logger.info("或手动下载模型：")
                logger.info("  方式1: huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints")
                logger.info("  方式2: git clone https://huggingface.co/IndexTeam/IndexTTS-2 checkpoints")
                raise
                
        except Exception as e:
            logger.error(f"下载模型失败: {str(e)}")
            raise
    
    def _load_official_inference(self):
        """加载官方推理接口"""
        try:
            # 尝试多种可能的导入方式
            inference_module = None
            inference_class = None
            
            # 方式1: 从官方仓库根目录导入
            try:
                from inference import IndexTTSInference
                inference_class = IndexTTSInference
                logger.debug("从 inference 模块导入成功")
            except ImportError:
                # 方式2: 从官方仓库子目录导入
                try:
                    sys.path.insert(0, str(self.official_repo_path))
                    from index_tts.inference import IndexTTSInference
                    inference_class = IndexTTSInference
                    logger.debug("从 index_tts.inference 导入成功")
                except ImportError:
                    # 方式3: 直接导入（如果代码在仓库根目录）
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            "inference",
                            self.official_repo_path / "inference.py"
                        )
                        inference_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(inference_module)
                        inference_class = inference_module.IndexTTSInference
                        logger.debug("从文件直接导入成功")
                    except Exception as e:
                        logger.error(f"所有导入方式都失败: {e}")
                        raise ImportError("无法导入官方推理模块")
            
            # 初始化推理器
            # 尝试不同的初始化参数组合
            init_params = {
                'checkpoint_dir': str(self.model_dir),
                'device': self.device
            }
            
            # 添加可选的config_path
            config_path = self.model_dir / 'config.yaml'
            if config_path.exists():
                init_params['config_path'] = str(config_path)
            
            try:
                self.inference = inference_class(**init_params)
            except TypeError:
                # 如果参数不匹配，尝试不传config_path
                init_params.pop('config_path', None)
                self.inference = inference_class(**init_params)
            
            logger.info("✓ 官方推理接口加载成功")
            
        except ImportError as e:
            logger.error(f"导入官方模块失败: {str(e)}")
            logger.error("请确保已正确安装官方代码:")
            logger.error("  1. 运行: git clone https://github.com/index-tts/index-tts.git")
            logger.error("  2. 或运行: bash src/tts/setup_indextts2.sh")
            raise
        except Exception as e:
            logger.error(f"加载推理接口失败: {str(e)}")
            logger.error(f"模型目录: {self.model_dir}")
            logger.error(f"官方代码目录: {self.official_repo_path}")
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
            
            # 准备参考音频
            if reference_audio_path is None and reference_audio is not None:
                # 保存临时文件
                import tempfile
                import soundfile as sf
                
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, reference_audio, self.sample_rate)
                reference_audio_path = temp_file.name
            
            # 调用官方推理接口
            # 根据官方API的实际接口调整参数
            synth_params = {
                'text': text,
            }
            
            # 添加参考音频（如果提供）
            if reference_audio_path:
                synth_params['reference_audio'] = reference_audio_path
            
            # 添加情感参数
            if emotion:
                synth_params['emotion'] = emotion
                synth_params['emotion_strength'] = emotion_strength
            
            # 添加语速
            if speed and speed != 1.0:
                synth_params['speed'] = speed
            
            # 添加其他参数
            synth_params.update(kwargs)
            
            # 调用官方接口（可能需要调整参数名）
            try:
                audio = self.inference.synthesize(**synth_params)
            except TypeError as e:
                # 如果参数不匹配，尝试简化参数
                logger.warning(f"完整参数调用失败: {e}，尝试简化参数")
                # 只传递必需参数
                minimal_params = {'text': text}
                if reference_audio_path:
                    minimal_params['reference_audio'] = reference_audio_path
                audio = self.inference.synthesize(**minimal_params)
            
            # 清理临时文件
            if 'temp_file' in locals():
                os.unlink(temp_file.name)
            
            logger.info(f"合成成功，音频长度: {len(audio)/self.sample_rate:.2f}秒")
            return audio
            
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
class IndexTTS2OfficialWrapper(IndexTTS2Official):
    """
    改进的官方模型包装器
    提供与复现模型完全兼容的接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 音频处理器（兼容接口）
        self.audio_processor = AudioProcessorWrapper(self.sample_rate)
        
        # 文本处理器（兼容接口）
        self.text_tokenizer = TextTokenizerWrapper()
        
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
IndexTTSModule = IndexTTS2OfficialWrapper


class AudioProcessorWrapper:
    """音频处理器包装器"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    def load_audio(self, path: str) -> np.ndarray:
        """加载音频"""
        import librosa
        audio, _ = librosa.load(path, sr=self.sample_rate)
        return audio
    
    def save_audio(self, audio: np.ndarray, path: str):
        """保存音频"""
        import soundfile as sf
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, audio, self.sample_rate)


class TextTokenizerWrapper:
    """文本处理器包装器"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本"""
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()