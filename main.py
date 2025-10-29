"""
智能学伴系统 - 主程序入口
实现：语音输入 -> ASR -> LLM -> TTS -> 语音输出
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from config import load_config
from src.pipeline import ConversationPipeline


def setup_logging(config):
    """配置日志系统"""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'data/logs/system.log')
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def interactive_mode(pipeline: ConversationPipeline):
    """交互式模式：手动输入音频文件路径进行对话"""
    print("\n" + "="*60)
    print("智能学伴系统 - 交互式模式")
    print("="*60)
    print("输入音频文件路径进行对话，输入 'quit' 退出，'reset' 重置对话")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("\n请输入音频文件路径 (或命令): ").strip()
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if user_input.lower() == 'reset':
                pipeline.reset_conversation()
                print("✓ 对话已重置")
                continue
            
            if not user_input:
                continue
            
            # 检查文件是否存在
            if not os.path.exists(user_input):
                print(f"✗ 文件不存在: {user_input}")
                continue
            
            # 处理音频
            print(f"\n处理中...")
            result = pipeline.process_audio_file(user_input)
            
            if result['success']:
                print(f"\n{'─'*60}")
                print(f"👤 用户: {result['asr_text']}")
                print(f"🤖 助手: {result['llm_response']}")
                print(f"{'─'*60}")
                if result.get('output_audio_path'):
                    print(f"🔊 语音已保存: {result['output_audio_path']}")
                print(f"✓ 完成 (第{pipeline.get_conversation_count()}轮对话)\n")
            else:
                print(f"\n✗ 处理失败: {result.get('error', '未知错误')}\n")
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n✗ 错误: {str(e)}\n")


def batch_mode(pipeline: ConversationPipeline, input_dir: str):
    """批处理模式：处理目录中的所有音频文件"""
    print("\n" + "="*60)
    print("智能学伴系统 - 批处理模式")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print("="*60 + "\n")
    
    # 支持的音频格式
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    # 查找所有音频文件
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    if not audio_files:
        print(f"✗ 未找到音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件\n")
    
    success_count = 0
    fail_count = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] 处理: {audio_file.name}")
        
        try:
            result = pipeline.process_audio_file(str(audio_file))
            
            if result['success']:
                print(f"  ✓ 成功")
                print(f"  用户: {result['asr_text']}")
                print(f"  助手: {result['llm_response'][:100]}...")
                success_count += 1
            else:
                print(f"  ✗ 失败: {result.get('error', '未知错误')}")
                fail_count += 1
        
        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"批处理完成: 成功 {success_count}, 失败 {fail_count}")
    print(f"{'='*60}\n")


def single_file_mode(pipeline: ConversationPipeline, audio_file: str):
    """单文件模式：处理单个音频文件"""
    print("\n" + "="*60)
    print("智能学伴系统 - 单文件模式")
    print("="*60)
    print(f"输入文件: {audio_file}")
    print("="*60 + "\n")
    
    if not os.path.exists(audio_file):
        print(f"✗ 文件不存在: {audio_file}")
        return
    
    print("处理中...\n")
    result = pipeline.process_audio_file(audio_file)
    
    if result['success']:
        print(f"{'─'*60}")
        print(f"👤 用户: {result['asr_text']}")
        print(f"🤖 助手: {result['llm_response']}")
        print(f"{'─'*60}")
        if result.get('output_audio_path'):
            print(f"🔊 语音已保存: {result['output_audio_path']}")
        print(f"\n✓ 处理成功\n")
    else:
        print(f"✗ 处理失败: {result.get('error', '未知错误')}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能学伴系统 - 语音对话助手')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'batch', 'single'],
        default='interactive',
        help='运行模式: interactive(交互式), batch(批处理), single(单文件)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='输入文件或目录路径（batch和single模式需要）'
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        print(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 设置日志
        logger = setup_logging(config)
        logger.info("="*60)
        logger.info("智能学伴系统启动")
        logger.info("="*60)
        
        # 初始化对话流程
        pipeline = ConversationPipeline(config)
        
        # 根据模式运行
        if args.mode == 'interactive':
            interactive_mode(pipeline)
        
        elif args.mode == 'batch':
            if not args.input:
                print("✗ 批处理模式需要指定 --input 目录")
                return
            batch_mode(pipeline, args.input)
        
        elif args.mode == 'single':
            if not args.input:
                print("✗ 单文件模式需要指定 --input 文件路径")
                return
            single_file_mode(pipeline, args.input)
        
        logger.info("智能学伴系统已退出")
        
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

