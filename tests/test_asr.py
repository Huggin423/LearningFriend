"""
ASR模块测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src.asr import FunASRModule


class TestFunASR:
    """FunASR模块测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        return load_config(str(config_path))
    
    @pytest.fixture
    def asr_module(self, config):
        """创建ASR模块实例"""
        asr_config = config.get('asr', {})
        return FunASRModule(asr_config)
    
    def test_module_init(self, asr_module):
        """测试模块初始化"""
        assert asr_module is not None
        assert asr_module.model is not None
        assert asr_module.sample_rate == 16000
    
    def test_transcribe_file(self, asr_module, tmp_path):
        """测试文件识别"""
        # 创建一个测试音频文件（静音）
        import soundfile as sf
        
        audio_data = np.zeros(16000, dtype=np.float32)  # 1秒静音
        test_file = tmp_path / "test.wav"
        sf.write(test_file, audio_data, 16000)
        
        # 测试识别（静音可能返回空或噪声识别结果）
        result = asr_module.transcribe_file(str(test_file))
        assert isinstance(result, str)
    
    def test_transcribe_array(self, asr_module):
        """测试数组识别"""
        # 创建测试音频数组
        audio_array = np.zeros(16000, dtype=np.float32)
        
        result = asr_module.transcribe_array(audio_array)
        assert isinstance(result, str)
    
    def test_set_hotword(self, asr_module):
        """测试设置热词"""
        hotword = "测试 热词"
        asr_module.set_hotword(hotword)
        assert asr_module.hotword == hotword


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

