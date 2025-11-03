"""
对话流程控制模块
整合ASR、LLM、TTS，实现完整的对话流程
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import soundfile as sf

from src.asr import FunASRModule
from src.llm import LLMInterface
from src.tts import IndexTTSModule


logger = logging.getLogger(__name__)


class ConversationPipeline:
    """对话流程控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话流程控制器
        
        Args:
            config: 完整的配置字典
        """
        self.config = config
        self.conversation_config = config.get('conversation', {})
        
        # 配置参数
        self.max_history = self.conversation_config.get('max_history', 10)
        self.save_audio = self.conversation_config.get('save_audio', True)
        self.audio_output_dir = self.conversation_config.get('audio_output_dir', 'data/audio_output')
        self.audio_input_dir = self.conversation_config.get('audio_input_dir', 'data/audio_input')
        
        # 创建必要的目录
        os.makedirs(self.audio_output_dir, exist_ok=True)
        os.makedirs(self.audio_input_dir, exist_ok=True)
        
        # 初始化各个模块
        logger.info("正在初始化对话流程控制器...")
        self.asr = FunASRModule(config.get('asr', {}))
        self.llm = LLMInterface(config.get('llm', {}))
        self.tts = IndexTTSModule(config.get('tts', {}))
        
        # 对话计数器
        self.conversation_count = 0
        
        logger.info("对话流程控制器初始化完成")
    
    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        处理音频文件，完成一轮完整对话
        
        Args:
            audio_path: 输入音频文件路径
            
        Returns:
            包含各个步骤结果的字典
        """
        try:
            logger.info(f"开始处理音频文件: {audio_path}")
            result = {
                'success': False,
                'input_audio': audio_path,
                'asr_text': None,
                'llm_response': None,
                'tts_audio': None,
                'output_audio_path': None,
                'error': None
            }
            
            # Step 1: ASR - 语音转文字
            logger.info("Step 1/3: ASR语音识别...")
            try:
                asr_text = self.asr.transcribe_file(audio_path)
                result['asr_text'] = asr_text
                logger.info(f"ASR识别结果: {asr_text}")
            except Exception as e:
                error_msg = f"ASR识别失败: {str(e)}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
            
            # Step 2: LLM - 生成回复
            logger.info("Step 2/3: LLM生成回复...")
            try:
                llm_response = self.llm.chat(asr_text, use_history=True)
                result['llm_response'] = llm_response
                logger.info(f"LLM回复: {llm_response}")
                
                # 修剪历史记录
                self.llm.trim_history(self.max_history)
            except Exception as e:
                error_msg = f"LLM生成失败: {str(e)}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
            
            # Step 3: TTS - 文字转语音
            logger.info("Step 3/3: TTS语音合成...")
            try:
                tts_audio = self.tts.synthesize(llm_response)
                result['tts_audio'] = tts_audio
                
                # 保存音频文件
                if self.save_audio:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"response_{timestamp}_{self.conversation_count:04d}.wav"
                    output_path = os.path.join(self.audio_output_dir, output_filename)
                    
                    sf.write(output_path, tts_audio, self.tts.sample_rate)
                    result['output_audio_path'] = output_path
                    logger.info(f"合成音频已保存: {output_path}")
                
            except Exception as e:
                error_msg = f"TTS合成失败: {str(e)}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
            
            # 成功完成
            self.conversation_count += 1
            result['success'] = True
            logger.info(f"对话处理完成 (第{self.conversation_count}轮)")
            
            return result
            
        except Exception as e:
            logger.error(f"处理对话时出错: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_audio_array(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        处理音频数组，完成一轮完整对话
        
        Args:
            audio_array: 音频数据数组
            sample_rate: 采样率
            
        Returns:
            包含各个步骤结果的字典
        """
        try:
            # 如果需要保存输入音频，先保存
            if self.save_audio:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                input_filename = f"input_{timestamp}_{self.conversation_count:04d}.wav"
                input_path = os.path.join(self.audio_input_dir, input_filename)
                sf.write(input_path, audio_array, sample_rate)
                logger.info(f"输入音频已保存: {input_path}")
            else:
                input_path = None
            
            result = {
                'success': False,
                'input_audio': input_path,
                'asr_text': None,
                'llm_response': None,
                'tts_audio': None,
                'output_audio_path': None,
                'error': None
            }
            
            # Step 1: ASR
            logger.info("Step 1/3: ASR语音识别...")
            try:
                asr_text = self.asr.transcribe_array(audio_array, sample_rate)
                result['asr_text'] = asr_text
                logger.info(f"ASR识别结果: {asr_text}")
            except Exception as e:
                error_msg = f"ASR识别失败: {str(e)}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
            
            # Step 2: LLM
            logger.info("Step 2/3: LLM生成回复...")
            try:
                llm_response = self.llm.chat(asr_text, use_history=True)
                result['llm_response'] = llm_response
                logger.info(f"LLM回复: {llm_response}")
                
                self.llm.trim_history(self.max_history)
            except Exception as e:
                error_msg = f"LLM生成失败: {str(e)}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
            
            # Step 3: TTS
            logger.info("Step 3/3: TTS语音合成...")
            try:
                # 获取默认参考音频路径（如果配置中有）
                tts_config = self.config.get('tts', {})
                default_ref_audio = tts_config.get('default_reference_audio')
                
                # 处理相对路径
                if default_ref_audio and not os.path.isabs(default_ref_audio):
                    from pathlib import Path
                    project_root = Path(__file__).parent.parent.parent
                    default_ref_audio = str(project_root / default_ref_audio)
                
                # 使用默认参考音频（如果可用）
                ref_audio_path = default_ref_audio if (default_ref_audio and os.path.exists(default_ref_audio)) else None
                tts_audio = self.tts.synthesize(
                    llm_response,
                    reference_audio_path=ref_audio_path
                )
                result['tts_audio'] = tts_audio
                
                if self.save_audio:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"response_{timestamp}_{self.conversation_count:04d}.wav"
                    output_path = os.path.join(self.audio_output_dir, output_filename)
                    
                    sf.write(output_path, tts_audio, self.tts.sample_rate)
                    result['output_audio_path'] = output_path
                    logger.info(f"合成音频已保存: {output_path}")
                
            except Exception as e:
                error_msg = f"TTS合成失败: {str(e)}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
            
            self.conversation_count += 1
            result['success'] = True
            logger.info(f"对话处理完成 (第{self.conversation_count}轮)")
            
            return result
            
        except Exception as e:
            logger.error(f"处理对话时出错: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def reset_conversation(self):
        """重置对话状态"""
        self.llm.clear_history()
        self.conversation_count = 0
        logger.info("对话状态已重置")
    
    def get_conversation_count(self) -> int:
        """获取对话轮数"""
        return self.conversation_count
    
    def get_history(self):
        """获取对话历史"""
        return self.llm.get_history()
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.llm.set_system_prompt(prompt)
    
    def set_speaker(self, speaker_id: int):
        """设置TTS音色"""
        self.tts.set_speaker(speaker_id)
    
    def set_speed(self, speed: float):
        """设置TTS语速"""
        self.tts.set_speed(speed)

