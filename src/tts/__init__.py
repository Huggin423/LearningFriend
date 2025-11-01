"""
TTS模块 - 文本转语音
支持官方模型和复现模型两种实现
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_tts_module(config: Dict[str, Any]):
    """
    创建TTS模块工厂函数
    根据配置选择使用官方模型或复现模型
    
    Args:
        config: TTS配置字典
        
    Returns:
        TTS模块实例
    """
    use_official = config.get('use_official', True)  # 默认使用官方模型
    
    if use_official:
        logger.info("使用IndexTTS2官方模型")
        try:
            from .indextts2_official_wrapper import IndexTTS2OfficialWrapper
            return IndexTTS2OfficialWrapper(config)
        except ImportError as e:
            logger.warning(f"官方模型加载失败: {e}，回退到复现模型")
            # 回退到复现模型
            from .indextts_module import IndexTTS2Reimplement
            return IndexTTS2Reimplement(config)
        except Exception as e:
            logger.warning(f"官方模型初始化失败: {e}，回退到复现模型")
            from .indextts_module import IndexTTS2Reimplement
            return IndexTTS2Reimplement(config)
    else:
        logger.info("使用IndexTTS2复现模型")
        from .indextts_module import IndexTTS2Reimplement
        return IndexTTS2Reimplement(config)


# 为了向后兼容，提供默认导出
# 实际的模块实例通过工厂函数创建
__all__ = ['create_tts_module']

# 为了兼容旧的导入方式，提供一个智能类
# 这个类会根据配置自动选择使用哪个实现
class IndexTTSModule:
    """
    兼容性别名类
    自动根据配置选择官方模型或复现模型
    """
    
    def __new__(cls, config: Dict[str, Any]):
        """工厂模式：根据配置创建对应实例"""
        return create_tts_module(config)
    
    # 为了IDE自动补全，提供静态方法提示
    @staticmethod
    def create(config: Dict[str, Any]):
        """显式创建方法"""
        return create_tts_module(config)

