"""
音频处理工具模块
提供音频读写、特征提取、预处理等功能
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa未安装，部分音频功能将不可用")


class AudioProcessor:
    """音频处理器"""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        n_fft: int = 1024,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.n_fft = n_fft
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax if mel_fmax is not None else sample_rate // 2
        
        if not HAS_LIBROSA:
            logger.error("需要librosa库进行音频处理")
            raise ImportError("请安装librosa: pip install librosa")
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
        
        Returns:
            audio: 音频波形数据
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def save_audio(self, audio: np.ndarray, file_path: str, sample_rate: Optional[int] = None):
        """
        保存音频文件
        
        Args:
            audio: 音频波形数据
            file_path: 保存路径
            sample_rate: 采样率（默认使用初始化值）
        """
        import soundfile as sf
        
        sr = sample_rate if sample_rate else self.sample_rate
        sf.write(file_path, audio, sr)
    
    def audio_to_mel(
        self,
        audio: np.ndarray,
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        将音频转换为Mel频谱
        
        Args:
            audio: 音频波形 [T]
            return_tensor: 是否返回torch.Tensor
        
        Returns:
            mel: Mel频谱 [n_mel_channels, frames]
        """
        # 使用librosa计算Mel频谱
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )
        
        # 转换为对数刻度
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # 归一化到[-1, 1]
        mel = mel / 80.0
        mel = np.clip(mel, -1.0, 1.0)
        
        if return_tensor:
            return torch.FloatTensor(mel)
        
        return mel
    
    def mel_to_audio(
        self,
        mel: Union[np.ndarray, torch.Tensor],
        griffin_lim_iters: int = 60
    ) -> np.ndarray:
        """
        从Mel频谱重建音频（使用Griffin-Lim）
        
        Args:
            mel: Mel频谱 [n_mel_channels, frames]
            griffin_lim_iters: Griffin-Lim迭代次数
        
        Returns:
            audio: 音频波形 [T]
        """
        if isinstance(mel, torch.Tensor):
            mel = mel.cpu().numpy()
        
        # 反归一化
        mel = mel * 80.0
        
        # 逆对数刻度
        mel = librosa.db_to_power(mel)
        
        # 使用Griffin-Lim重建波形
        audio = librosa.feature.inverse.mel_to_audio(
            M=mel,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            n_iter=griffin_lim_iters
        )
        
        return audio
    
    def adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        调整音频速度
        
        Args:
            audio: 音频波形 [T]
            speed: 速度倍率 (>1加快, <1减慢)
        
        Returns:
            adjusted_audio: 调整后的音频
        """
        import soundfile as sf
        import io
        
        if speed == 1.0:
            return audio
        
        # 使用librosa的time_stretch
        audio_stretched = librosa.effects.time_stretch(audio, rate=1.0/speed)
        
        return audio_stretched
    
    def adjust_pitch(self, audio: np.ndarray, pitch_shift: float) -> np.ndarray:
        """
        调整音高
        
        Args:
            audio: 音频波形 [T]
            pitch_shift: 音高偏移（半音数）
        
        Returns:
            adjusted_audio: 调整后的音频
        """
        if pitch_shift == 0:
            return audio
        
        # 使用librosa的音高偏移
        audio_shifted = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=pitch_shift
        )
        
        return audio_shifted
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        音频归一化
        
        Args:
            audio: 音频波形 [T]
            target_db: 目标分贝数
        
        Returns:
            normalized_audio: 归一化后的音频
        """
        # 计算RMS
        rms = librosa.feature.rms(y=audio)[0]
        mean_rms = np.mean(rms)
        
        if mean_rms > 0:
            # 计算目标RMS
            target_rms = 10 ** (target_db / 20)
            
            # 归一化
            normalized_audio = audio * (target_rms / mean_rms)
        else:
            normalized_audio = audio
        
        return normalized_audio
    
    def trim_silence(
        self,
        audio: np.ndarray,
        top_db: float = 60.0
    ) -> np.ndarray:
        """
        去除静音
        
        Args:
            audio: 音频波形 [T]
            top_db: 阈值（分贝）
        
        Returns:
            trimmed_audio: 修剪后的音频
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def split_audio(
        self,
        audio: np.ndarray,
        duration: float,
        overlap: float = 0.0
    ) -> list:
        """
        分割音频为固定时长的片段
        
        Args:
            audio: 音频波形 [T]
            duration: 每段时长（秒）
            overlap: 重叠时长（秒）
        
        Returns:
            segments: 音频片段列表
        """
        frame_length = int(duration * self.sample_rate)
        frame_step = int((duration - overlap) * self.sample_rate)
        
        segments = []
        for i in range(0, len(audio) - frame_length + 1, frame_step):
            segment = audio[i:i + frame_length]
            segments.append(segment)
        
        # 处理最后一段
        if len(audio) - i > frame_step:
            segment = audio[i:]
            # 用零填充到指定长度
            padding = np.zeros(frame_length - len(segment))
            segment = np.concatenate([segment, padding])
            segments.append(segment)
        
        return segments
    
    def resample(self, audio: np.ndarray, target_sr: int) -> np.ndarray:
        """
        重采样音频
        
        Args:
            audio: 音频波形 [T]
            target_sr: 目标采样率
        
        Returns:
            resampled_audio: 重采样后的音频
        """
        if target_sr == self.sample_rate:
            return audio
        
        resampled = librosa.resample(audio, self.sample_rate, target_sr)
        return resampled


class AudioEncoder:
    """音频编码器（用于提取语义特征等）"""
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 256):
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """
        提取音频特征
        
        Args:
            audio: 音频波形 [T]
        
        Returns:
            features: 特征字典
        """
        if not HAS_LIBROSA:
            return {}
        
        # MFCC特征
        mfccs = librosa.feature.mfcc(y=audio, n_mfcc=13)
        
        # 频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(y=audio)[0]
        
        # 过零率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # 色度特征
        chroma = librosa.feature.chroma_stft(y=audio)
        
        return {
            'mfccs': mfccs,
            'spectral_centroids': spectral_centroids,
            'zero_crossing_rate': zero_crossing_rate,
            'chroma': chroma
        }
