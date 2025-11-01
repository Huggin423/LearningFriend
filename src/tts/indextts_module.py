"""
IndexTTS2语音合成模块
基于IndexTTS2论文的完整实现
支持情感控制、精确时长控制和零样本语音合成
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from .models.text_to_semantic import TextToSemanticModule
from .models.semantic_to_mel import SemanticToMelModule
from .models.vocoder import BigVGANv2Vocoder
from .models.text_to_emotion import TextToEmotionModule
from .utils.audio_utils import AudioProcessor
from .utils.text_utils import TextTokenizer

logger = logging.getLogger(__name__)


class IndexTTS2Reimplement:
    """IndexTTS2复现模型（论文实现）"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化IndexTTS2模块
        
        Args:
            config: TTS配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 模型路径配置
        self.model_dir = Path(config.get('model_path', 'models/indextts2'))
        self.t2s_checkpoint = self.model_dir / config.get('t2s_checkpoint', 't2s_model.pth')
        self.s2m_checkpoint = self.model_dir / config.get('s2m_checkpoint', 's2m_model.pth')
        self.vocoder_checkpoint = self.model_dir / config.get('vocoder_checkpoint', 'vocoder.pth')
        self.t2e_checkpoint = self.model_dir / config.get('t2e_checkpoint', 't2e_model.pth')
        
        # 音频参数
        self.sample_rate = config.get('sample_rate', 22050)
        self.hop_length = config.get('hop_length', 256)
        self.n_mel_channels = config.get('n_mel_channels', 80)
        
        # 合成参数
        self.speaker_id = config.get('speaker_id', 0)
        self.speed = config.get('speed', 1.0)
        self.emotion = config.get('emotion', 'neutral')
        
        # 初始化组件
        logger.info("正在初始化IndexTTS2模块...")
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化所有子模块"""
        try:
            # 音频处理器
            self.audio_processor = AudioProcessor(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                n_mel_channels=self.n_mel_channels
            )
            
            # 文本分词器
            self.text_tokenizer = TextTokenizer(
                vocab_size=self.config.get('vocab_size', 4096)
            )
            
            # Text-to-Semantic 模块
            self.t2s_module = TextToSemanticModule(
                vocab_size=self.text_tokenizer.vocab_size,
                d_model=self.config.get('t2s_d_model', 512),
                n_heads=self.config.get('t2s_n_heads', 8),
                n_layers=self.config.get('t2s_n_layers', 12),
                max_seq_length=self.config.get('max_seq_length', 2048),
                device=self.device
            )
            
            # Semantic-to-Mel 模块
            self.s2m_module = SemanticToMelModule(
                n_mel_channels=self.n_mel_channels,
                d_model=self.config.get('s2m_d_model', 512),
                device=self.device
            )
            
            # Vocoder 模块
            self.vocoder = BigVGANv2Vocoder(
                n_mel_channels=self.n_mel_channels,
                sample_rate=self.sample_rate,
                device=self.device
            )
            
            # Text-to-Emotion 模块（可选）
            if self.config.get('enable_emotion_control', True):
                self.t2e_module = TextToEmotionModule(
                    model_name=self.config.get('t2e_model', 'Qwen-3-1.7b'),
                    device=self.device
                )
            else:
                self.t2e_module = None
            
            # 加载模型权重
            self._load_checkpoints()
            
            logger.info("IndexTTS2模块初始化成功")
            logger.info(f"设备: {self.device}")
            
        except Exception as e:
            logger.error(f"初始化IndexTTS2失败: {str(e)}")
            raise
    
    def _load_checkpoints(self):
        """加载模型检查点"""
        if self.t2s_checkpoint.exists():
            logger.info(f"加载T2S模型: {self.t2s_checkpoint}")
            self.t2s_module.load_state_dict(torch.load(self.t2s_checkpoint, map_location=self.device))
        else:
            logger.warning(f"T2S检查点未找到: {self.t2s_checkpoint}，使用随机初始化")
            
        if self.s2m_checkpoint.exists():
            logger.info(f"加载S2M模型: {self.s2m_checkpoint}")
            self.s2m_module.load_state_dict(torch.load(self.s2m_checkpoint, map_location=self.device))
        else:
            logger.warning(f"S2M检查点未找到: {self.s2m_checkpoint}，使用随机初始化")
            
        if self.vocoder_checkpoint.exists():
            logger.info(f"加载Vocoder: {self.vocoder_checkpoint}")
            self.vocoder.load_state_dict(torch.load(self.vocoder_checkpoint, map_location=self.device))
        else:
            logger.warning(f"Vocoder检查点未找到: {self.vocoder_checkpoint}，使用随机初始化")
            
        if self.t2e_module is not None and self.t2e_checkpoint.exists():
            logger.info(f"加载T2E模型: {self.t2e_checkpoint}")
            self.t2e_module.load_lora_weights(self.t2e_checkpoint)
    
    def synthesize(
        self,
        text: str,
        timbre_prompt: Optional[np.ndarray] = None,
        style_prompt: Optional[np.ndarray] = None,
        emotion: Optional[str] = None,
        target_duration: Optional[float] = None,
        speed: Optional[float] = None,
        speaker_id: Optional[int] = None
    ) -> np.ndarray:
        """
        文本转语音合成
        
        Args:
            text: 要合成的文本
            timbre_prompt: 音色参考音频 (可选)
            style_prompt: 风格/情感参考音频 (可选)
            emotion: 情感描述或标签 (可选)
            target_duration: 目标时长(秒) (可选，用于精确时长控制)
            speed: 语速倍率 (可选)
            speaker_id: 说话人ID (可选)
            
        Returns:
            音频数组 (numpy.ndarray)
        """
        try:
            logger.info(f"开始合成: {text[:50]}...")
            
            # 参数处理
            if speed is None:
                speed = self.speed
            if speaker_id is None:
                speaker_id = self.speaker_id
            if emotion is None:
                emotion = self.emotion
            
            # 1. 文本编码
            text_tokens = self.text_tokenizer.encode(text)
            text_tokens = torch.LongTensor(text_tokens).unsqueeze(0).to(self.device)
            
            # 2. 提取说话人特征
            speaker_embedding = self._extract_speaker_embedding(
                timbre_prompt, speaker_id
            )
            
            # 3. 提取情感特征
            emotion_embedding = self._extract_emotion_embedding(
                style_prompt, emotion, text
            )
            
            # 4. 计算目标token数量（用于时长控制）
            target_token_num = None
            if target_duration is not None:
                target_token_num = self._duration_to_tokens(target_duration)
            
            # 5. Text-to-Semantic: 生成语义token
            with torch.no_grad():
                semantic_tokens, gpt_latents = self.t2s_module.generate(
                    text_tokens=text_tokens,
                    speaker_embedding=speaker_embedding,
                    emotion_embedding=emotion_embedding,
                    target_token_num=target_token_num
                )
            
            logger.debug(f"生成语义tokens: {semantic_tokens.shape}")
            
            # 6. Semantic-to-Mel: 生成梅尔频谱
            with torch.no_grad():
                mel_spectrogram = self.s2m_module.generate(
                    semantic_tokens=semantic_tokens,
                    semantic_embedding_weight=self.t2s_module.semantic_embedding.weight,
                    speaker_embedding=speaker_embedding,
                    gpt_latents=gpt_latents,
                    timbre_prompt=timbre_prompt
                )
            
            logger.debug(f"生成梅尔频谱: {mel_spectrogram.shape}")
            
            # 7. Vocoder: 梅尔频谱转音频波形
            with torch.no_grad():
                audio = self.vocoder.generate(mel_spectrogram)
            
            # 转换为numpy数组
            audio = audio.squeeze().cpu().numpy()
            
            # 8. 应用语速调整
            if speed != 1.0:
                audio = self.audio_processor.adjust_speed(audio, speed)
            
            logger.info(f"合成完成，音频时长: {len(audio)/self.sample_rate:.2f}秒")
            return audio
            
        except Exception as e:
            logger.error(f"语音合成失败: {str(e)}")
            raise
    
    def _extract_speaker_embedding(
        self,
        timbre_prompt: Optional[np.ndarray],
        speaker_id: int
    ) -> torch.Tensor:
        """提取说话人嵌入向量"""
        if timbre_prompt is not None:
            # 从参考音频提取说话人特征
            mel = self.audio_processor.audio_to_mel(timbre_prompt)
            mel = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            speaker_embedding = self.t2s_module.speaker_encoder(mel)
        else:
            # 使用预定义的说话人ID
            speaker_embedding = self.t2s_module.get_speaker_embedding(speaker_id)
        
        return speaker_embedding
    
    def _extract_emotion_embedding(
        self,
        style_prompt: Optional[np.ndarray],
        emotion: str,
        text: str
    ) -> torch.Tensor:
        """提取情感嵌入向量"""
        if style_prompt is not None:
            # 从参考音频提取情感特征
            mel = self.audio_processor.audio_to_mel(style_prompt)
            mel = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            emotion_embedding = self.t2s_module.emotion_encoder(mel)
        elif self.t2e_module is not None and emotion:
            # 使用T2E模块从文本生成情感向量
            emotion_probs = self.t2e_module.predict_emotion(text if emotion == 'auto' else emotion)
            emotion_embedding = self.t2e_module.get_emotion_embedding(emotion_probs)
            emotion_embedding = emotion_embedding.to(self.device)
        else:
            # 使用默认中性情感
            emotion_embedding = self.t2s_module.get_emotion_embedding('neutral')
        
        return emotion_embedding
    
    def _duration_to_tokens(self, duration: float) -> int:
        """将时长(秒)转换为语义token数量"""
        # 根据经验值估算：约每秒20-25个tokens
        tokens_per_second = 22
        return int(duration * tokens_per_second)
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        **kwargs
    ) -> str:
        """
        文本转语音并保存到文件
        
        Args:
            text: 要合成的文本
            output_path: 输出音频文件路径
            **kwargs: 其他参数，传递给synthesize()
            
        Returns:
            输出文件路径
        """
        import soundfile as sf
        
        # 生成音频
        audio = self.synthesize(text, **kwargs)
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        
        logger.info(f"音频已保存到: {output_path}")
        return output_path
    
    def synthesize_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """
        批量合成
        
        Args:
            texts: 文本列表
            **kwargs: 其他参数
            
        Returns:
            音频数组列表
        """
        audios = []
        for i, text in enumerate(texts):
            logger.info(f"合成进度: {i+1}/{len(texts)}")
            audio = self.synthesize(text, **kwargs)
            audios.append(audio)
        return audios
    
    def set_speaker(self, speaker_id: int):
        """设置说话人ID"""
        self.speaker_id = speaker_id
        logger.info(f"设置说话人ID: {speaker_id}")
    
    def set_speed(self, speed: float):
        """设置语速"""
        self.speed = max(0.5, min(2.0, speed))
        logger.info(f"设置语速: {self.speed}")
    
    def set_emotion(self, emotion: str):
        """设置默认情感"""
        self.emotion = emotion
        logger.info(f"设置情感: {emotion}")
    
    def clone_voice(
        self,
        reference_audio: np.ndarray,
        text: str,
        **kwargs
    ) -> np.ndarray:
        """
        零样本语音克隆
        
        Args:
            reference_audio: 参考音频（用于提取音色）
            text: 要合成的文本
            **kwargs: 其他参数
            
        Returns:
            合成的音频
        """
        return self.synthesize(
            text=text,
            timbre_prompt=reference_audio,
            **kwargs
        )
    
    def __del__(self):
        """析构函数"""
        logger.debug("IndexTTS2模块已卸载")