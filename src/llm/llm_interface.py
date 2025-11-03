"""
LLM接口模块
支持DeepSeek和Qwen等商业API调用
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI


logger = logging.getLogger(__name__)


class LLMInterface:
    """LLM接口，支持多种商业API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM接口
        
        Args:
            config: LLM配置字典
        """
        self.config = config
        self.provider = config.get('provider', 'deepseek')
        self.system_prompt = config.get('system_prompt', '你是一个友好的智能助手。')
        
        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []
        
        # 初始化客户端
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化API客户端"""
        try:
            if self.provider == 'deepseek':
                provider_config = self.config.get('deepseek', {})
                api_key = provider_config.get('api_key', '')
                base_url = provider_config.get('base_url', 'https://api.deepseek.com')
                
                if not api_key:
                    logger.warning("DeepSeek API Key未配置！请在config.yaml中填写")
                
                # 验证并清理 base_url（OpenAI客户端会自动添加 /chat/completions）
                if '/chat/completions' in base_url:
                    logger.warning(
                        f"检测到 base_url 包含 '/chat/completions'，这将导致404错误。"
                        f"已将 base_url 从 '{base_url}' 修正为正确的格式。"
                    )
                    base_url = base_url.replace('/chat/completions', '').rstrip('/')
                    if not base_url.endswith('/v1'):
                        base_url = base_url.rstrip('/') + '/v1'
                
                # DeepSeek使用OpenAI兼容接口
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                self.model_name = provider_config.get('model', 'deepseek-chat')
                self.temperature = provider_config.get('temperature', 0.7)
                self.max_tokens = provider_config.get('max_tokens', 2000)
                self.top_p = provider_config.get('top_p', 0.95)
                
                logger.info(f"DeepSeek客户端初始化成功，模型: {self.model_name}")
            
            elif self.provider == 'qwen':
                provider_config = self.config.get('qwen', {})
                api_key = provider_config.get('api_key', '')
                base_url = provider_config.get('base_url', 'https://dashscope.aliyuncs.com/api/v1')
                
                if not api_key:
                    logger.warning("Qwen API Key未配置！请在config.yaml中填写")
                
                # Qwen也使用OpenAI兼容接口
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                self.model_name = provider_config.get('model', 'qwen-turbo')
                self.temperature = provider_config.get('temperature', 0.7)
                self.max_tokens = provider_config.get('max_tokens', 2000)
                self.top_p = provider_config.get('top_p', 0.8)
                
                logger.info(f"Qwen客户端初始化成功，模型: {self.model_name}")
            
            else:
                raise ValueError(f"不支持的LLM提供商: {self.provider}")
        
        except Exception as e:
            logger.error(f"初始化LLM客户端失败: {str(e)}")
            raise
    
    def chat(self, user_message: str, use_history: bool = True) -> str:
        """
        与LLM对话
        
        Args:
            user_message: 用户消息
            use_history: 是否使用对话历史
            
        Returns:
            LLM的回复
        """
        if self.client is None:
            raise RuntimeError("LLM客户端未初始化")
        
        try:
            # 构建消息列表
            messages = []
            
            # 添加系统提示词
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            
            # 添加历史对话
            if use_history and self.conversation_history:
                messages.extend(self.conversation_history)
            
            # 添加当前用户消息
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            logger.debug(f"发送消息到{self.provider}: {user_message}")
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            
            # 提取回复
            assistant_message = response.choices[0].message.content
            
            # 更新对话历史
            if use_history:
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
            
            logger.info(f"LLM回复: {assistant_message[:100]}...")
            return assistant_message
        
        except Exception as e:
            logger.error(f"LLM对话失败: {str(e)}")
            raise
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history.copy()
    
    def set_history(self, history: List[Dict[str, str]]):
        """设置对话历史"""
        self.conversation_history = history
        logger.info(f"已设置对话历史，共{len(history)}条消息")
    
    def trim_history(self, max_turns: int = 10):
        """
        修剪对话历史，保留最近的N轮对话
        
        Args:
            max_turns: 最大保留轮数
        """
        if len(self.conversation_history) > max_turns * 2:
            self.conversation_history = self.conversation_history[-(max_turns * 2):]
            logger.info(f"对话历史已修剪至{max_turns}轮")
    
    def set_system_prompt(self, prompt: str):
        """
        设置系统提示词
        
        Args:
            prompt: 新的系统提示词
        """
        self.system_prompt = prompt
        logger.info("系统提示词已更新")
    
    def switch_provider(self, provider: str):
        """
        切换LLM提供商
        
        Args:
            provider: 提供商名称 (deepseek/qwen)
        """
        if provider not in ['deepseek', 'qwen']:
            raise ValueError(f"不支持的提供商: {provider}")
        
        self.provider = provider
        self._init_client()
        logger.info(f"已切换到提供商: {provider}")

