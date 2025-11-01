"""
BigVGANv2 Vocoder Module
高保真音频波形生成模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ResidualStack(nn.Module):
    """残差堆栈"""
    def __init__(self, channels: int):
        super().__init__()
        self.stack = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 3, padding=1)
        )
    
    def forward(self, x):
        return x + self.stack(x)


class BigVGANv2Generator(nn.Module):
    """BigVGANv2 生成器"""
    
    def __init__(
        self,
        n_mel_channels: int = 80,
        channels: int = 512,
        kernel_size: int = 7,
        stride: int = 4,
        n_upsamples: int = 4
    ):
        super().__init__()
        self.n_upsamples = n_upsamples
        self.stride = stride
        self.kernel_size = kernel_size
        
        # 初始卷积
        self.input_conv = nn.Conv1d(
            n_mel_channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # 上采样 + 残差块
        self.upsampling_blocks = nn.ModuleList()
        current_channels = channels
        current_stride = 1
        
        for i in range(n_upsamples):
            # 转置卷积上采样
            up_block = nn.ModuleList([
                nn.ConvTranspose1d(
                    current_channels,
                    current_channels // 2,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2,
                    output_padding=stride % 2
                ),
                nn.GroupNorm(1, current_channels // 2),
                nn.ReLU()
            ])
            self.upsampling_blocks.append(up_block)
            
            # 残差块
            res_block = ResidualStack(current_channels // 2)
            self.upsampling_blocks.append(res_block)
            
            current_channels = current_channels // 2
            current_stride *= stride
        
        # 最终卷积
        self.output_conv = nn.Sequential(
            nn.Conv1d(current_channels, current_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(current_channels, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel频谱 [B, n_mel_channels, F]
        
        Returns:
            audio: 音频波形 [B, 1, T]
        """
        x = self.input_conv(mel)
        
        # 上采样块
        idx = 0
        while idx < len(self.upsampling_blocks):
            # 上采样
            for layer in self.upsampling_blocks[idx]:
                x = layer(x)
            idx += 1
            
            # 残差
            x = self.upsampling_blocks[idx](x)
            idx += 1
        
        # 输出
        audio = self.output_conv(x)
        
        return audio


class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器"""
    
    def __init__(self):
        super().__init__()
        
        # 不同尺度的判别器
        self.discriminators = nn.ModuleList([
            self._build_discriminator(),  # 原始尺度
            self._build_discriminator(),  # 下采样2倍
            self._build_discriminator(),  # 下采样4倍
        ])
    
    def _build_discriminator(self):
        """构建单个判别器"""
        layers = []
        in_channels = 1
        
        for out_channels in [32, 64, 128, 256, 512]:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=1, padding=7),
                nn.LeakyReLU(0.2),
                nn.Conv1d(out_channels, out_channels, kernel_size=41, stride=4, groups=4, padding=20),
                nn.GroupNorm(4, out_channels),
                nn.LeakyReLU(0.2)
            ])
            in_channels = out_channels
        
        layers.append(nn.Conv1d(in_channels, 1, kernel_size=3, padding=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, audio: torch.Tensor):
        """
        Args:
            audio: 音频波形 [B, 1, T]
        
        Returns:
            scores: 各尺度的判别分数 [N, B, 1, T']
        """
        scores = []
        
        for disc in self.discriminators:
            score = disc(audio)
            scores.append(score)
            audio = F.avg_pool1d(audio, kernel_size=2, stride=2)
        
        return scores


class BigVGANv2Vocoder(nn.Module):
    """BigVGANv2 Vocoder模块"""
    
    def __init__(
        self,
        n_mel_channels: int = 80,
        sample_rate: int = 22050,
        device: str = 'cuda'
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sample_rate = sample_rate
        self.device = device
        
        # 生成器
        self.generator = BigVGANv2Generator(
            n_mel_channels=n_mel_channels,
            channels=512,
            kernel_size=7,
            stride=4,
            n_upsamples=4
        )
        
        # 判别器（训练时使用，推理时不需要）
        self.discriminator = MultiScaleDiscriminator()
        
        self.to(device)
        self.eval()  # 默认推理模式
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        推理生成音频
        
        Args:
            mel: Mel频谱 [B, n_mel_channels, F]
        
        Returns:
            audio: 音频波形 [B, T]
        """
        # Mel转音频
        audio = self.generator(mel)  # [B, 1, T]
        audio = audio.squeeze(1)  # [B, T]
        
        return audio
    
    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        """
        生成音频（推理接口）
        
        Args:
            mel: Mel频谱 [B, F, n_mel_channels] 或 [B, n_mel_channels, F]
        
        Returns:
            audio: 音频波形 [B, T]
        """
        self.eval()
        
        # 确保Mel格式正确 [B, n_mel_channels, F]
        if mel.dim() == 3 and mel.shape[-1] == self.n_mel_channels:
            mel = mel.transpose(1, 2)  # [B, F, n_mel] -> [B, n_mel, F]
        
        with torch.no_grad():
            audio = self.generator(mel)
            audio = audio.squeeze(1)  # [B, 1, T] -> [B, T]
        
        return audio
