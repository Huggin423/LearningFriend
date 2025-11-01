"""
Text-to-Semantic (T2S) Module
基于Transformer的自回归语义令牌生成模块
实现论文中的时长控制和情感控制功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GradientReversalLayer(nn.Module):
    """梯度反转层，用于对抗训练解耦情感和说话人特征"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return x
    
    def backward(self, grad_output):
        return -self.alpha * grad_output


class SpeakerConditioner(nn.Module):
    """说话人感知条件器，从音频中提取音色特征"""
    
    def __init__(self, d_model: int = 512, n_mel_channels: int = 80):
        super().__init__()
        self.d_model = d_model
        
        # Conformer-based encoder
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(n_mel_channels, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.speaker_projection = nn.Linear(d_model, d_model)
        self.n_speakers = 100  # 预定义的说话人数量
        
        # 说话人嵌入层
        self.speaker_embeddings = nn.Embedding(self.n_speakers, d_model)
    
    def forward(self, mel: torch.Tensor, speaker_id: Optional[torch.Tensor] = None):
        """
        Args:
            mel: Mel频谱 [B, n_mel_channels, T]
            speaker_id: 说话人ID [B]
        Returns:
            speaker_embedding: 说话人嵌入 [B, d_model]
        """
        if speaker_id is not None:
            # 使用预定义的说话人嵌入
            return self.speaker_embeddings(speaker_id)
        
        # 从Mel频谱提取说话人特征
        batch_size = mel.size(0)
        features = self.mel_encoder(mel)  # [B, d_model, 1]
        features = features.squeeze(-1)   # [B, d_model]
        speaker_embedding = self.speaker_projection(features)  # [B, d_model]
        
        return speaker_embedding
    
    def get_speaker_embedding(self, speaker_id: int) -> torch.Tensor:
        """获取指定说话人的嵌入"""
        return self.speaker_embeddings(torch.LongTensor([speaker_id])).squeeze(0)


class EmotionConditioner(nn.Module):
    """情感感知条件器，从音频中提取情感特征"""
    
    def __init__(self, d_model: int = 512, n_mel_channels: int = 80):
        super().__init__()
        self.d_model = d_model
        
        # Conformer-based encoder (简化版，使用CNN)
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(n_mel_channels, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.emotion_projection = nn.Linear(d_model, d_model)
        
        # 预定义的情感嵌入
        self.emotion_embeddings = nn.Embedding(7, d_model)  # 7种基本情感
    
    def forward(self, mel: torch.Tensor):
        """
        Args:
            mel: Mel频谱 [B, n_mel_channels, T]
        Returns:
            emotion_embedding: 情感嵌入 [B, d_model]
        """
        features = self.mel_encoder(mel)  # [B, d_model, 1]
        features = features.squeeze(-1)   # [B, d_model]
        emotion_embedding = self.emotion_projection(features)  # [B, d_model]
        return emotion_embedding
    
    def get_emotion_embedding(self, emotion: str) -> torch.Tensor:
        """获取指定情感的嵌入"""
        emotion_map = {
            'neutral': 0, 'happiness': 1, 'sadness': 2,
            'anger': 3, 'fear': 4, 'disgust': 5, 'surprise': 6
        }
        idx = emotion_map.get(emotion, 0)
        return self.emotion_embeddings(torch.LongTensor([idx])).squeeze(0)


class TextToSemanticModule(nn.Module):
    """Text-to-Semantic模块：自回归生成语义令牌"""
    
    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        max_seq_length: int = 2048,
        device: str = 'cuda'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.max_seq_length = max_seq_length
        
        # 文本嵌入
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        # 语义令牌嵌入
        self.semantic_embedding = nn.Embedding(vocab_size, d_model)
        
        # 说话人条件器
        self.speaker_conditioner = SpeakerConditioner(d_model)
        
        # 情感条件器
        self.emotion_conditioner = EmotionConditioner(d_model)
        
        # 边界令牌
        self.bt_token = nn.Parameter(torch.randn(1, d_model))  # Begin Text
        self.ba_token = nn.Parameter(torch.randn(1, d_model))  # Begin Audio
        self.ea_token = nn.Parameter(torch.randn(1, d_model))  # End Audio
        
        # 时长控制嵌入表
        max_tokens = 2000  # 最大token数
        self.duration_embedding = nn.Embedding(max_tokens, d_model)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, d_model)
        )
        
        # 说话人分类器（用于对抗训练）
        self.speaker_classifier = nn.Linear(d_model, self.speaker_conditioner.n_speakers)
    
    def forward(
        self,
        text_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        emotion_embedding: Optional[torch.Tensor] = None,
        target_token_num: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            text_tokens: 文本令牌 [B, T_text]
            semantic_tokens: 语义令牌 [B, T_sem]
            speaker_embedding: 说话人嵌入 [B, d_model]
            emotion_embedding: 情感嵌入 [B, d_model] (可选)
            target_token_num: 目标语义令牌数量 (可选)
        Returns:
            logits: 输出logits [B, T_sem, vocab_size]
        """
        batch_size = text_tokens.size(0)
        
        # 构建输入序列
        seq_parts = []
        
        # 1. 说话人和情感嵌入
        if emotion_embedding is not None:
            combined = speaker_embedding + emotion_embedding  # [B, d_model]
        else:
            combined = speaker_embedding
        seq_parts.append(combined.unsqueeze(1))  # [B, 1, d_model]
        
        # 2. 时长控制嵌入
        if target_token_num is not None:
            dur_token = self.duration_embedding(
                torch.clamp(torch.LongTensor([target_token_num]), 0, self.duration_embedding.num_embeddings - 1)
            ).unsqueeze(0).to(self.device)
        else:
            # 设为0表示自由生成
            dur_token = torch.zeros(1, 1, self.d_model).to(self.device)
        
        seq_parts.append(dur_token.expand(batch_size, -1, -1))
        
        # 3. 边界令牌BT
        seq_parts.append(self.bt_token.expand(batch_size, -1, -1))
        
        # 4. 文本嵌入
        text_emb = self.text_embedding(text_tokens)  # [B, T_text, d_model]
        seq_parts.append(text_emb)
        
        # 5. 边界令牌BA
        seq_parts.append(self.ba_token.expand(batch_size, -1, -1))
        
        # 6. 语义令牌嵌入
        semantic_emb = self.semantic_embedding(semantic_tokens)  # [B, T_sem, d_model]
        seq_parts.append(semantic_emb)
        
        # 拼接整个序列
        input_seq = torch.cat(seq_parts, dim=1)  # [B, L, d_model]
        
        # 添加位置编码
        seq_len = input_seq.size(1)
        if seq_len <= self.max_seq_length:
            input_seq = input_seq + self.positional_encoding[:, :seq_len, :]
        
        # Transformer解码
        tgt = input_seq
        memory = input_seq
        
        output = self.transformer(tgt, memory)  # [B, L, d_model]
        
        # 只对语义令牌部分取输出
        text_len = text_tokens.size(1)
        semantic_start_idx = 3 + text_len + 1  # c + p + BT + text + BA
        semantic_output = output[:, semantic_start_idx:, :]  # [B, T_sem, d_model]
        
        # 输出投影
        logits = self.output_projection(semantic_output)  # [B, T_sem, vocab_size]
        
        return logits
    
    def generate(
        self,
        text_tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        emotion_embedding: Optional[torch.Tensor] = None,
        target_token_num: Optional[int] = None,
        max_length: int = 1000,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自回归生成语义令牌
        
        Args:
            text_tokens: 文本令牌 [1, T_text]
            speaker_embedding: 说话人嵌入 [1, d_model]
            emotion_embedding: 情感嵌入 [1, d_model] (可选)
            target_token_num: 目标令牌数量 (可选)
            max_length: 最大生成长度
            temperature: 采样温度
        
        Returns:
            semantic_tokens: 生成的语义令牌 [1, T_gen]
            gpt_latents: GPT隐层特征 [1, T_gen, d_model]
        """
        self.eval()
        batch_size = 1
        
        # 初始化序列
        seq_parts = []
        combined = speaker_embedding + (emotion_embedding if emotion_embedding is not None else 0)
        seq_parts.append(combined.unsqueeze(1))
        
        # 时长控制
        if target_token_num is not None:
            dur_token = self.duration_embedding(
                torch.clamp(torch.LongTensor([target_token_num]), 0, self.duration_embedding.num_embeddings - 1)
            ).unsqueeze(0).to(self.device)
        else:
            dur_token = torch.zeros(1, 1, self.d_model).to(self.device)
        seq_parts.append(dur_token)
        
        seq_parts.append(self.bt_token.unsqueeze(0))
        text_emb = self.text_embedding(text_tokens)
        seq_parts.append(text_emb)
        seq_parts.append(self.ba_token.unsqueeze(0))
        
        input_seq = torch.cat(seq_parts, dim=1)
        
        # 生成语义令牌
        generated_tokens = []
        current_length = input_seq.size(1)
        
        # 添加位置编码
        if current_length <= self.max_seq_length:
            input_seq = input_seq + self.positional_encoding[:, :current_length, :]
        
        # 自回归生成
        for step in range(max_length):
            # Transformer解码
            tgt = input_seq
            memory = input_seq
            output = self.transformer(tgt, memory)
            
            # 获取最后一个位置的输出
            last_output = output[:, -1:, :]  # [1, 1, d_model]
            
            # 保存GPT隐层特征
            gpt_latents = last_output if step == 0 else torch.cat([gpt_latents, last_output], dim=1)
            
            # 预测下一个token
            logits = self.output_projection(last_output)  # [1, 1, vocab_size]
            logits = logits / temperature
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1)  # [1, 1]
            
            generated_tokens.append(next_token)
            
            # 添加到序列
            next_emb = self.semantic_embedding(next_token)
            input_seq = torch.cat([input_seq, next_emb], dim=1)
            current_length += 1
            
            # 检查是否结束（简化，使用特殊token或长度限制）
            if next_token.item() == 0:  # 假设0是结束符
                break
        
        semantic_tokens = torch.cat(generated_tokens, dim=1)  # [1, T_gen]
        
        return semantic_tokens, gpt_latents

