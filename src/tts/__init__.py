"""
TTS模块 - 文本转语音
使用 ModelScope 官方 IndexTTS2 模型
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_tts_module(config: Dict[str, Any]):
    """
    创建TTS模块工厂函数
    使用 ModelScope 官方 IndexTTS2 模型
    
    Args:
        config: TTS配置字典
        
    Returns:
        TTS模块实例
    """
    logger.info("使用 IndexTTS2 ModelScope 官方模型")
    
    try:
        from .indextts2_modelscope import IndexTTS2ModelScopeWrapper
        return IndexTTS2ModelScopeWrapper(config)
    except ImportError as e:
        logger.error(f"ModelScope未安装: {e}")
        logger.error("请运行: pip install modelscope")
        raise
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        raise


# 为了向后兼容，提供默认导出
# 实际的模块实例通过工厂函数创建
__all__ = ['create_tts_module']

# 为了兼容旧的导入方式，提供一个智能类
class IndexTTSModule:
    """
    兼容性别名类
    使用 ModelScope 官方模型
    """
    
    def __new__(cls, config: Dict[str, Any]):
        """工厂模式：根据配置创建对应实例"""
        return create_tts_module(config)
    
    # 为了IDE自动补全，提供静态方法提示
    @staticmethod
    def create(config: Dict[str, Any]):
        """显式创建方法"""
        return create_tts_module(config)

