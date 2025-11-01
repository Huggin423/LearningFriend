"""
文本处理工具模块
提供分词、规范化、音素转换等功能
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
import torch

logger = logging.getLogger(__name__)


class TextNormalizer:
    """文本规范化器"""
    
    def __init__(self):
        # 数字到中文的映射
        self.number_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        
        # 标点符号映射
        self.punctuation_map = {
            ',': '，', '.': '。', '!': '！', '?': '？',
            ';': '；', ':': '：', '(': '（', ')': '）'
        }
    
    def normalize(self, text: str) -> str:
        """
        规范化文本
        
        Args:
            text: 输入文本
        
        Returns:
            normalized_text: 规范化后的文本
        """
        # 全角转半角
        text = self._to_halfwidth(text)
        
        # 数字转中文
        text = self._number_to_chinese(text)
        
        # 标点符号规范化
        text = self._normalize_punctuation(text)
        
        # 去除多余空格
        text = ' '.join(text.split())
        
        return text
    
    def _to_halfwidth(self, text: str) -> str:
        """全角转半角"""
        def _convert_char(char):
            if '\uff01' <= char <= '\uff60':
                return chr(ord(char) - 0xfee0)
            return char
        
        return ''.join(_convert_char(c) for c in text)
    
    def _number_to_chinese(self, text: str) -> str:
        """将数字转换为中文"""
        def _replace_number(match):
            num_str = match.group()
            return ''.join(self.number_map.get(d, d) for d in num_str)
        
        return re.sub(r'\d+', _replace_number, text)
    
    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点符号"""
        for eng, chn in self.punctuation_map.items():
            text = text.replace(eng, chn)
        return text


class PhonemConverter:
    """音素转换器"""
    
    def __init__(self):
        # 中文声母
        self.initials = [
            'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
            'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch',
            'sh', 'r', 'z', 'c', 's', 'y', 'w'
        ]
        
        # 中文韵母（简化）
        self.finals = [
            'a', 'o', 'e', 'i', 'u', 'v',
            'ai', 'ei', 'ao', 'ou', 'ia', 'ie', 'ua', 'uo', 've',
            'iao', 'iou', 'uai', 'uei', 'an', 'en', 'ian', 'in',
            'uan', 'uen', 'van', 'vn', 'ang', 'eng', 'iang', 'ing',
            'uang', 'ueng', 'ong', 'iong'
        ]
    
    def convert_to_phonemes(self, text: str) -> List[str]:
        """
        将文本转换为音素序列
        
        Args:
            text: 输入文本
        
        Returns:
            phonemes: 音素列表
        """
        # 简化实现：按字符分割
        # 实际应该使用专业的中文拼音转换工具
        phonemes = []
        for char in text:
            if char.isalnum():
                # 这里应该调用实际的音素转换
                phonemes.append(char)
            else:
                phonemes.append(' ')
        
        return phonemes


class TextTokenizer:
    """文本分词器"""
    
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        
        # 特殊token
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.mask_token = '<mask>'
        
        # 构建词表
        self._build_vocab()
        
        # 文本规范化器
        self.normalizer = TextNormalizer()
        
        # 音素转换器
        self.phonem_converter = PhonemConverter()
    
    def _build_vocab(self):
        """构建词表"""
        # 特殊tokens
        special_tokens = [
            self.pad_token, self.unk_token, self.bos_token,
            self.eos_token, self.mask_token
        ]
        
        # 字符tokens（简化，实际应该有更复杂的词表）
        char_tokens = []
        
        # ASCII字符
        for i in range(32, 127):
            char_tokens.append(chr(i))
        
        # 常用中文字符（简化）
        common_chinese = [
            '的', '一', '是', '在', '了', '不', '和', '有', '大',
            '这', '主', '人', '中', '上', '为', '们', '年', '个'
        ]
        char_tokens.extend(common_chinese)
        
        # 拼音字母
        for c in 'abcdefghijklmnopqrstuvwxyz':
            char_tokens.append(c.upper())
        
        # 数字
        for d in '0123456789':
            char_tokens.append(d)
        
        # 组合词表
        self.vocab_list = special_tokens + char_tokens
        
        # 截断到vocab_size
        if len(self.vocab_list) > self.vocab_size:
            self.vocab_list = self.vocab_list[:self.vocab_size]
        else:
            # 补充空白token
            num_missing = self.vocab_size - len(self.vocab_list)
            self.vocab_list.extend([f'<unused{i}>' for i in range(num_missing)])
        
        # 创建token到id和id到token的映射
        self.token_to_id = {token: i for i, token in enumerate(self.vocab_list)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab_list)}
        
        # 更新词汇大小
        self.vocab_size = len(self.vocab_list)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True
    ) -> List[int]:
        """
        将文本编码为token ids
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token
            max_length: 最大长度
            padding: 是否填充
            truncation: 是否截断
        
        Returns:
            token_ids: token id列表
        """
        # 规范化文本
        normalized_text = self.normalizer.normalize(text)
        
        # 分词（简化：按字符）
        tokens = list(normalized_text)
        
        # 转换为ids
        token_ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id[self.unk_token])
            token_ids.append(token_id)
        
        # 添加特殊token
        if add_special_tokens:
            token_ids = [self.token_to_id[self.bos_token]] + token_ids + [self.token_to_id[self.eos_token]]
        
        # 截断
        if max_length is not None and truncation and len(token_ids) > max_length:
            if add_special_tokens:
                # 保留BOS和EOS
                token_ids = token_ids[:max_length-1] + [self.token_to_id[self.eos_token]]
            else:
                token_ids = token_ids[:max_length]
        
        # 填充
        if max_length is not None and padding and len(token_ids) < max_length:
            pad_id = self.token_to_id[self.pad_token]
            token_ids = token_ids + [pad_id] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        将token ids解码为文本
        
        Args:
            token_ids: token id列表
            skip_special_tokens: 是否跳过特殊token
        
        Returns:
            text: 解码后的文本
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id < len(self.vocab_list):
                token = self.id_to_token[token_id]
                
                if skip_special_tokens and token in [
                    self.pad_token, self.unk_token,
                    self.bos_token, self.eos_token, self.mask_token
                ]:
                    continue
                
                tokens.append(token)
        
        text = ''.join(tokens)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
        
        Returns:
            tokens: token列表
        """
        normalized_text = self.normalizer.normalize(text)
        return list(normalized_text)
    
    def get_vocab(self) -> Dict[str, int]:
        """获取词表"""
        return self.token_to_id.copy()


def create_tokenizer(vocab_path: Optional[str] = None) -> TextTokenizer:
    """
    创建分词器
    
    Args:
        vocab_path: 词表路径（可选）
    
    Returns:
        tokenizer: 分词器实例
    """
    if vocab_path:
        # 从文件加载词表
        logger.warning("从文件加载词表功能未实现")
    
    # 创建默认分词器
    return TextTokenizer()

