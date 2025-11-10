"""
API工具函数
"""

import os
import logging
import base64
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def decode_base64_audio(base64_data: str, suffix: str = '.wav') -> str:
    """
    解码Base64音频数据并保存为临时文件
    
    Args:
        base64_data: Base64编码的音频数据
        suffix: 文件后缀
    
    Returns:
        临时文件路径
    """
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]
        
        audio_bytes = base64.decode(base64_data)
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        )
        temp_file.write(audio_bytes)
        temp_file.close()
        
        logger.debug(f"已解码Base64音频并保存到: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"解码Base64音频失败: {str(e)}")
        raise ValueError(f"无效的Base64音频数据: {str(e)}")


def cleanup_temp_file(file_path: str):
    """
    清理临时文件
    
    Args:
        file_path: 文件路径
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"已清理临时文件: {file_path}")
    except Exception as e:
        logger.warning(f"清理临时文件失败: {str(e)}")


def validate_audio_file(file_path: str) -> bool:
    """
    验证音频文件是否存在且可读
    
    Args:
        file_path: 文件路径
    
    Returns:
        是否有效
    """
    if not file_path:
        return False
    
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"音频文件不存在: {file_path}")
        return False
    
    if not path.is_file():
        logger.warning(f"路径不是文件: {file_path}")
        return False

    valid_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']
    if path.suffix.lower() not in valid_extensions:
        logger.warning(f"不支持的音频格式: {path.suffix}")
        return False
    
    return True


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    import yaml
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise


def setup_logging(log_level: str = "INFO"):
    """
    配置日志
    
    Args:
        log_level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
