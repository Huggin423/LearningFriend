"""
Text-to-Emotion (T2E) Module
自然语言情感控制模块
使用LoRA知识蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """LoRA层"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1.0 / rank
    
    def forward(self, x: torch.Tensor):
        return x @ (self.lora_A @ self.lora_B) * self.scaling


class TextToEmotionModule(nn.Module):
    """Text-to-Emotion模块：从文本映射到情感概率"""
    
    def __init__(
        self,
        model_name: str = 'Qwen-3-1.7b',
        vocab_size: int = 151936,
        d_model: int = 2048,
        n_emotions: int = 7,
        rank: int = 16,
        device: str = 'cuda'
    ):
        super().__init__()
        self.model_name = model_name
        self.n_emotions = n_emotions
        self.device = device
        
        # 定义7种基本情感
        self.emotion_list = [
            'neutral', 'happiness', 'sadness',
            'anger', 'fear', 'disgust', 'surprise'
        ]
        
        # 简化的文本编码器（实际使用预训练的Qwen）
        # 这里用Embedding替代
        self.text_encoder = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
                num_layers=4
            )
        )
        
        # LoRA层用于微调
        self.lora_layers = nn.ModuleList([
            LoRALayer(d_model, d_model, rank) for _ in range(4)
        ])
        
        # 情感分类头
        self.emotion_head = nn.Linear(d_model, n_emotions)
        
        # 预定义的情感嵌入 (V集合)
        # 这些应该从真实情感音频中提取，这里用随机初始化
        self.emotion_embeddings = nn.Parameter(
            torch.randn(n_emotions, 10, d_model) * 0.02
        )
        
        self.to(device)
    
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            text_tokens: 文本令牌 [B, T]
        
        Returns:
            emotion_probs: 情感概率分布 [B, n_emotions]
        """
        # 文本编码
        x = self.text_encoder(text_tokens)  # [B, T, d_model]
        
        # 池化
        x = x.mean(dim=1)  # [B, d_model]
        
        # LoRA层
        for lora in self.lora_layers:
            x = x + lora(x)
        
        # 情感分类
        emotion_logits = self.emotion_head(x)  # [B, n_emotions]
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        return emotion_probs
    
    def predict_emotion(self, text: str) -> torch.Tensor:
        """
        从文本预测情感
        
        Args:
            text: 输入文本
        
        Returns:
            emotion_probs: 情感概率 [n_emotions]
        """
        # 简化：将文本转换为token（实际应该使用分词器）
        # 这里用简单的哈希映射
        text_tokens = self._text_to_tokens(text)
        text_tokens = torch.LongTensor(text_tokens).unsqueeze(0).to(self.device)
        
        self.eval()
        with torch.no_grad():
            emotion_probs = self.forward(text_tokens)  # [1, n_emotions]
        
        return emotion_probs.squeeze(0)  # [n_emotions]
    
    def get_emotion_embedding(self, emotion_probs: torch.Tensor) -> torch.Tensor:
        """
        从情感概率计算情感嵌入
        
        Args:
            emotion_probs: 情感概率 [n_emotions] 或 [B, n_emotions]
        
        Returns:
            emotion_embedding: 情感嵌入向量 [d_model] 或 [B, d_model]
        """
        if emotion_probs.dim() == 1:
            emotion_probs = emotion_probs.unsqueeze(0)
        
        batch_size = emotion_probs.size(0)
        
        # 加权平均公式: e_input = sum_e(p_e * mean(v_e))
        emotion_emb_list = []
        
        for e_idx in range(self.n_emotions):
            # V_e 中所有向量的平均
            v_e = self.emotion_embeddings[e_idx]  # [num_samples, d_model]
            v_e_mean = v_e.mean(dim=0)  # [d_model]
            
            # 加权
            weighted = emotion_probs[:, e_idx].unsqueeze(-1) * v_e_mean  # [B, d_model]
            emotion_emb_list.append(weighted)
        
        # 求和
        emotion_embedding = sum(emotion_emb_list)  # [B, d_model]
        
        if emotion_embedding.size(0) == 1:
            emotion_embedding = emotion_embedding.squeeze(0)
        
        return emotion_embedding
    
    def _text_to_tokens(self, text: str) -> List[int]:
        """
        将文本转换为token（简化版）
        实际应该使用完整的分词器
        
        Args:
            text: 输入文本
        
        Returns:
            tokens: token列表
        """
        # 简化实现：基于字符的哈希
        vocab_size = 151936
        
        tokens = []
        for char in text:
            token = hash(char) % vocab_size
            tokens.append(token)
        
        if len(tokens) == 0:
            tokens = [0]  # 至少一个token
        
        return tokens
    
    def load_lora_weights(self, checkpoint_path: str):
        """加载LoRA权重"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.lora_layers.load_state_dict(checkpoint['lora'])
            self.emotion_head.load_state_dict(checkpoint['emotion_head'])
            logger.info(f"LoRA权重已加载: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"加载LoRA权重失败: {str(e)}")
    
    def get_emotion_by_name(self, emotion_name: str) -> torch.Tensor:
        """
        根据情感名称获取对应的嵌入
        
        Args:
            emotion_name: 情感名称
        
        Returns:
            emotion_embedding: 情感嵌入 [d_model]
        """
        emotion_probs = torch.zeros(self.n_emotions).to(self.device)
        
        for idx, emo in enumerate(self.emotion_list):
            if emo == emotion_name.lower():
                emotion_probs[idx] = 1.0
                break
        else:
            # 如果未匹配，使用neutral
            emotion_probs[0] = 1.0
        
        return self.get_emotion_embedding(emotion_probs)

