"""
端到端测试脚本
测试语音输入到语音输出的完整流程
"""

import os
import sys
import logging
import numpy as np
import soundfile as sf

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from src.asr import FunASRModule
from src.llm import LLMInterface
from src.tts import IndexTTSModule


def test_asr_module():
    """测试ASR模块"""
    print("\n" + "="*60)
    print("测试1: ASR模块 - FunASR语音识别")
    print("="*60)
    
    try:
        config = load_config()
        asr = FunASRModule(config['asr'])
        print("✓ ASR模块初始化成功")
        print(f"  模型: {asr.model_name}")
        print(f"  设备: {asr.device}")
        print(f"  采样率: {asr.sample_rate}Hz")
        
        # 创建一个测试音频（静音用于测试）
        test_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
        
        # 测试识别
        print("\n尝试识别测试音频...")
        result = asr.transcribe_array(test_audio)
        print(f"✓ ASR识别完成: '{result}'")
        
        return True
        
    except Exception as e:
        print(f"✗ ASR模块测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_module():
    """测试LLM模块"""
    print("\n" + "="*60)
    print("测试2: LLM模块 - DeepSeek-V3对话")
    print("="*60)
    
    try:
        config = load_config()
        llm = LLMInterface(config['llm'])
        print("✓ LLM模块初始化成功")
        print(f"  提供商: {llm.provider}")
        print(f"  模型: {llm.model_name}")
        print(f"  基础URL: {llm.client.base_url}")
        
        # 测试对话（需要有效的API Key）
        api_key = config['llm']['deepseek'].get('api_key', '')
        if not api_key:
            print("⚠ 警告: API Key未配置，跳过LLM对话测试")
            print("  请在 config/config.yaml 中填入你的硅基流动 API Key")
            return None
        
        print("\n尝试发送测试消息...")
        response = llm.chat("你好，请简单介绍一下自己", use_history=False)
        print(f"✓ LLM回复: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM模块测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_module():
    """测试TTS模块"""
    print("\n" + "="*60)
    print("测试3: TTS模块 - IndexTTS2语音合成")
    print("="*60)
    
    try:
        config = load_config()
        tts = IndexTTSModule(config['tts'])
        print("✓ TTS模块初始化成功")
        print(f"  设备: {tts.device}")
        print(f"  采样率: {tts.sample_rate}Hz")
        print(f"  音色ID: {tts.speaker_id}")
        print(f"  语速: {tts.speed}")
        
        # 测试合成
        print("\n尝试合成语音...")
        test_text = "你好，我是智能学伴助手"
        audio = tts.synthesize(test_text)
        print(f"✓ TTS合成完成")
        print(f"  音频长度: {len(audio)} 样本")
        print(f"  音频时长: {len(audio)/tts.sample_rate:.2f} 秒")
        
        # 保存测试音频
        output_dir = "data/audio_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_tts_output.wav")
        sf.write(output_path, audio, tts.sample_rate)
        print(f"  已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ TTS模块测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试完整流程"""
    print("\n" + "="*60)
    print("测试4: 完整对话流程")
    print("="*60)
    
    try:
        from src.pipeline import ConversationPipeline
        
        config = load_config()
        pipeline = ConversationPipeline(config)
        print("✓ 对话流程控制器初始化成功")
        
        # 创建一个测试音频
        test_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
        
        print("\n尝试处理完整对话流程...")
        result = pipeline.process_audio_array(test_audio, sample_rate=16000)
        
        if result['success']:
            print("✓ 完整流程测试成功")
            print(f"  ASR识别: {result['asr_text']}")
            print(f"  LLM回复: {result['llm_response'][:100]}..." if len(result['llm_response']) > 100 else f"  LLM回复: {result['llm_response']}")
            if result.get('output_audio_path'):
                print(f"  输出音频: {result['output_audio_path']}")
            print(f"  对话轮数: {pipeline.get_conversation_count()}")
        else:
            print(f"✗ 完整流程测试失败: {result.get('error', '未知错误')}")
        
        return result['success']
        
    except Exception as e:
        print(f"✗ 完整流程测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("智能学伴系统 - 端到端测试")
    print("="*60)
    print("此测试将验证语音输入到语音输出的完整流程")
    print("="*60)
    
    # 设置日志
    logging.basicConfig(
        level=logging.WARNING,  # 只显示警告和错误
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    results = {}
    
    # 测试各个模块
    results['asr'] = test_asr_module()
    results['llm'] = test_llm_module()
    results['tts'] = test_tts_module()
    
    # 测试完整流程
    if results['asr'] and (results['llm'] is not False) and results['tts']:
        results['pipeline'] = test_full_pipeline()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for module, result in results.items():
        if result is True:
            status = "✓ 通过"
        elif result is False:
            status = "✗ 失败"
        elif result is None:
            status = "⚠ 跳过"
        else:
            status = "✗ 失败"
        print(f"  {module.upper():10s}: {status}")
    
    print("="*60)
    
    # 判断整体结果
    if all(r is True or r is None for r in results.values()):
        print("\n🎉 所有测试通过！系统可以正常运行。")
        print("\n下一步:")
        print("1. 确保在 config/config.yaml 中配置了有效的API Key")
        print("2. 准备一个测试音频文件")
        print("3. 运行: python main.py --mode interactive")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查配置和依赖。")
        return 1


if __name__ == '__main__':
    sys.exit(main())

