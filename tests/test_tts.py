"""
TTS模块测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src.tts import IndexTTSModule


class TestIndexTTS:
    """IndexTTS模块测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        return load_config(str(config_path))
    
    @pytest.fixture
    def tts_module(self, config):
        """创建TTS模块实例"""
        tts_config = config.get('tts', {})
        return IndexTTSModule(tts_config)
    
    def test_module_init(self, tts_module):
        """测试模块初始化"""
        assert tts_module is not None
        assert tts_module.sample_rate > 0
    
    def test_synthesize(self, tts_module):
        """测试语音合成"""
        text = "这是一个测试"
        audio = tts_module.synthesize(text)
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32
    
    def test_synthesize_to_file(self, tts_module, tmp_path):
        """测试合成到文件"""
        text = "测试音频文件"
        output_path = tmp_path / "test_output.wav"
        
        result_path = tts_module.synthesize_to_file(text, str(output_path))
        assert Path(result_path).exists()
    
    def test_set_speaker(self, tts_module):
        """测试设置音色"""
        tts_module.set_speaker(1)
        assert tts_module.speaker_id == 1
    
    def test_set_speed(self, tts_module):
        """测试设置语速"""
        tts_module.set_speed(1.5)
        assert tts_module.speed == 1.5
    
    def test_set_pitch(self, tts_module):
        """测试设置音高"""
        tts_module.set_pitch(1.2)
        assert tts_module.pitch == 1.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

