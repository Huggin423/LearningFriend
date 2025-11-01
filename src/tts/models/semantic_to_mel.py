"""
Semantic-to-Mel (S2M) Module
基于流匹配(Flow Matching)的非自回归Mel谱生成模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FlowMatchingS2M(nn.Module):
    """基于流匹配的S2M模块"""
    
    def __init__(
        self,
        n_mel_channels: int = 80,
        d_model: int = 512,
        n_flows: int = 8
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.d_model = d_model
        self.n_flows = n_flows
        
        # 语义特征投影
        self.semantic_projection = nn.Linear(d_model, d_model)
        
        # 说话人特征投影
        self.speaker_projection = nn.Linear(d_model, d_model)
        
        # GPT隐层投影（GPT Latent Enhancement）
        self.gpt_projection = nn.Linear(d_model, d_model)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Flow Matching网络
        self.flow_networks = nn.ModuleList([
            self._build_flow_network() for _ in range(n_flows)
        ])
        
        # 时间步嵌入
        self.time_embedding = nn.Embedding(1000, d_model)
        
        # Mel频谱重建头
        self.mel_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, n_mel_channels)
        )
    
    def _build_flow_network(self):
        """构建单个Flow网络"""
        return nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model * 2),  # +1 for time
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
    
    def forward(
        self,
        mel_target: torch.Tensor,
        semantic_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
        gpt_latents: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            mel_target: 目标Mel频谱 [B, F, n_mel_channels]
            semantic_features: 语义特征 [B, T, d_model]
            speaker_embedding: 说话人嵌入 [B, d_model]
            gpt_latents: GPT隐层特征 [B, T, d_model] (可选)
            timesteps: 时间步 [B]
        
        Returns:
            mel_pred: 预测的Mel频谱 [B, F, n_mel_channels]
        """
        batch_size, mel_len, _ = mel_target.shape
        _, sem_len, _ = semantic_features.shape
        
        # 投影语义特征
        sem_feat = self.semantic_projection(semantic_features)  # [B, T, d_model]
        
        # GPT Latent Enhancement (50%概率融合)
        if gpt_latents is not None and torch.rand(1).item() > 0.5:
            gpt_feat = self.gpt_projection(gpt_latents)  # [B, T, d_model]
            # 向量加法融合
            sem_feat = sem_feat + gpt_feat
        elif gpt_latents is not None:
            # MLP融合
            gpt_feat = self.gpt_projection(gpt_latents)
            combined = torch.cat([sem_feat, gpt_feat], dim=-1)  # [B, T, 2*d_model]
            sem_feat = self.fusion_mlp(combined)
        
        # 说话人特征
        spk_feat = self.speaker_projection(speaker_embedding)  # [B, d_model]
        spk_feat = spk_feat.unsqueeze(1).expand(-1, sem_len, -1)  # [B, T, d_model]
        
        # 拼接特征
        combined_feat = sem_feat + spk_feat  # [B, T, d_model]
        
        # 上采样到Mel长度
        if sem_len != mel_len:
            combined_feat = F.interpolate(
                combined_feat.transpose(1, 2),  # [B, d_model, T]
                size=mel_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [B, F, d_model]
        
        # 时间步嵌入
        if timesteps is None:
            timesteps = torch.randint(0, 1000, (batch_size,)).to(combined_feat.device)
        time_emb = self.time_embedding(timesteps)  # [B, d_model]
        time_emb = time_emb.unsqueeze(1).expand(-1, mel_len, -1)  # [B, F, d_model]
        
        # Flow Matching
        x = combined_feat
        for i, flow_net in enumerate(self.flow_networks):
            # 添加时间信息
            time_input = torch.rand(x.size(0), x.size(1), 1).to(x.device)
            x_with_time = torch.cat([x, time_input], dim=-1)  # [B, F, d_model+1]
            
            # Flow网络
            flow = flow_net(x_with_time)  # [B, F, d_model]
            
            # 更新
            x = x + flow * (1.0 / self.n_flows)
        
        # 解码为Mel频谱
        mel_pred = self.mel_decoder(x)  # [B, F, n_mel_channels]
        
        return mel_pred
    
    def generate(
        self,
        semantic_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
        gpt_latents: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ) -> torch.Tensor:
        """
        推理生成Mel频谱
        
        Args:
            semantic_features: 语义特征 [B, T, d_model]
            speaker_embedding: 说话人嵌入 [B, d_model]
            gpt_latents: GPT隐层特征 [B, T, d_model] (可选)
            n_steps: ODE求解步数
        
        Returns:
            mel_spectrogram: 生成的Mel频谱 [B, F, n_mel_channels]
        """
        self.eval()
        batch_size = semantic_features.size(0)
        _, sem_len, _ = semantic_features.shape
        
        # 投影语义特征
        sem_feat = self.semantic_projection(semantic_features)
        
        # GPT Latent Enhancement (推理时使用加法融合)
        if gpt_latents is not None:
            gpt_feat = self.gpt_projection(gpt_latents)
            sem_feat = sem_feat + gpt_feat
        
        # 说话人特征
        spk_feat = self.speaker_projection(speaker_embedding)
        spk_feat = spk_feat.unsqueeze(1).expand(-1, sem_len, -1)
        
        # 拼接
        combined_feat = sem_feat + spk_feat
        
        # 从高斯噪声开始
        x = torch.randn_like(combined_feat)
        mel_len = combined_feat.size(1)
        
        # ODE求解
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.ones(batch_size).to(x.device) * (step * dt)
            time_emb = self.time_embedding((t * 999).long())
            time_emb = time_emb.unsqueeze(1).expand(-1, mel_len, -1)
            
            # Flow网络
            for i, flow_net in enumerate(self.flow_networks):
                time_input = torch.ones(x.size(0), x.size(1), 1).to(x.device) * (step * dt)
                x_with_time = torch.cat([x, time_input], dim=-1)
                flow = flow_net(x_with_time)
                x = x + flow * dt / self.n_flows
        
        # 解码
        mel_spectrogram = self.mel_decoder(x)
        
        return mel_spectrogram


class SemanticToMelModule(nn.Module):
    """Semantic-to-Mel模块封装"""
    
    def __init__(
        self,
        n_mel_channels: int = 80,
        d_model: int = 512,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        self.flow_matching = FlowMatchingS2M(
            n_mel_channels=n_mel_channels,
            d_model=d_model
        )
        
        self.to(device)
    
    def forward(
        self,
        mel_target: torch.Tensor,
        semantic_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
        gpt_latents: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """训练前向传播"""
        return self.flow_matching(
            mel_target=mel_target,
            semantic_features=semantic_features,
            speaker_embedding=speaker_embedding,
            gpt_latents=gpt_latents
        )
    
    def generate(
        self,
        semantic_tokens: torch.Tensor,
        semantic_embedding_weight: torch.Tensor,
        speaker_embedding: torch.Tensor,
        gpt_latents: Optional[torch.Tensor] = None,
        timbre_prompt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        生成Mel频谱
        
        Args:
            semantic_tokens: 语义令牌 [1, T]
            semantic_embedding_weight: 语义嵌入权重表
            speaker_embedding: 说话人嵌入 [1, d_model]
            gpt_latents: GPT隐层特征 [1, T, d_model]
            timbre_prompt: 音色参考Mel [1, F, n_mel_channels] (可选)
        
        Returns:
            mel_spectrogram: Mel频谱 [1, F, n_mel_channels]
        """
        # 将语义令牌转换为嵌入
        semantic_features = F.embedding(semantic_tokens, semantic_embedding_weight)  # [1, T, d_model]
        
        # 生成Mel频谱
        mel_spectrogram = self.flow_matching.generate(
            semantic_features=semantic_features,
            speaker_embedding=speaker_embedding,
            gpt_latents=gpt_latents
        )
        
        return mel_spectrogram
