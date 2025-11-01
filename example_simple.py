"""
简单的使用示例
演示如何快速使用智能学伴系统
"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from src.pipeline import ConversationPipeline


def main():
    print("\n" + "="*60)
    print("智能学伴系统 - 简单示例")
    print("="*60)
    
    try:
        # 加载配置
        print("\n[1/3] 加载配置...")
        config = load_config()
        print("✓ 配置加载成功")
        
        # 初始化系统
        print("\n[2/3] 初始化对话流程控制器...")
        pipeline = ConversationPipeline(config)
        print("✓ 系统初始化成功")
        
        # 创建一个简单的测试音频（静音，用于演示）
        print("\n[3/3] 准备测试音频...")
        test_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
        print("✓ 测试音频已创建")
        
        # 处理对话
        print("\n" + "-"*60)
        print("开始处理对话...")
        print("-"*60 + "\n")
        
        result = pipeline.process_audio_array(test_audio, sample_rate=16000)
        
        # 显示结果
        if result['success']:
            print("\n" + "="*60)
            print("对话处理成功！")
            print("="*60)
            print(f"\n用户输入 (ASR识别):")
            print(f"  {result['asr_text']}")
            print(f"\n助手回复 (LLM生成):")
            print(f"  {result['llm_response']}")
            
            if result.get('output_audio_path'):
                print(f"\n输出音频文件:")
                print(f"  {result['output_audio_path']}")
            
            print(f"\n对话统计:")
            print(f"  总轮数: {pipeline.get_conversation_count()}")
            print(f"  历史长度: {len(pipeline.get_history())}")
            
        else:
            print("\n" + "="*60)
            print("对话处理失败")
            print("="*60)
            print(f"\n错误信息:")
            print(f"  {result.get('error', '未知错误')}")
        
        print("\n" + "="*60)
        print("完成")
        print("="*60)
        print("\n提示:")
        print("- 这是静音音频的演示，实际使用时请提供真实的语音文件")
        print("- 运行 'python main.py --mode interactive' 进行交互式对话")
        print("- 运行 'python test_pipeline.py' 进行完整测试")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

