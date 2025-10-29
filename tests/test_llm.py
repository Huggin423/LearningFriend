"""
LLM模块测试
"""

import pytest
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src.llm import LLMInterface


class TestLLM:
    """LLM模块测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        return load_config(str(config_path))
    
    @pytest.fixture
    def llm_module(self, config):
        """创建LLM模块实例"""
        llm_config = config.get('llm', {})
        return LLMInterface(llm_config)
    
    def test_module_init(self, llm_module):
        """测试模块初始化"""
        assert llm_module is not None
        assert llm_module.client is not None
        assert llm_module.provider in ['deepseek', 'qwen']
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-api-tests", default=False),
        reason="需要 --run-api-tests 标志才运行API测试"
    )
    def test_chat(self, llm_module):
        """测试对话功能（需要有效的API Key）"""
        response = llm_module.chat("你好")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_history_management(self, llm_module):
        """测试历史管理"""
        # 清空历史
        llm_module.clear_history()
        assert len(llm_module.get_history()) == 0
        
        # 设置历史
        test_history = [
            {"role": "user", "content": "测试1"},
            {"role": "assistant", "content": "回复1"}
        ]
        llm_module.set_history(test_history)
        assert len(llm_module.get_history()) == 2
        
        # 测试修剪
        llm_module.trim_history(max_turns=1)
        assert len(llm_module.get_history()) == 2  # 1轮 = 2条消息
    
    def test_set_system_prompt(self, llm_module):
        """测试设置系统提示词"""
        new_prompt = "你是一个测试助手"
        llm_module.set_system_prompt(new_prompt)
        assert llm_module.system_prompt == new_prompt
    
    def test_switch_provider(self, llm_module, config):
        """测试切换提供商"""
        original_provider = llm_module.provider
        
        # 切换到另一个提供商
        new_provider = 'qwen' if original_provider == 'deepseek' else 'deepseek'
        llm_module.switch_provider(new_provider)
        assert llm_module.provider == new_provider
        
        # 切换回来
        llm_module.switch_provider(original_provider)
        assert llm_module.provider == original_provider


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

