#!/usr/bin/env python3
"""
性能基准测试脚本
测试整个流程的运行时间，包括：
1. 各模块初始化时间
2. 各模块单次处理时间
3. 端到端流程时间
4. 多次运行的平均时间
"""

import os
import sys
import time
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_config():
    """加载配置"""
    from config import load_config as _load_config
    return _load_config()

def find_reference_audio():
    """查找参考音频文件"""
    possible_paths = [
        project_root / "index-tts" / "examples" / "test_voice.wav",
        project_root / "data" / "audio_input" / "input_20251103_110735_0000.wav",
        project_root / "index-tts" / "examples" / "voice_01.wav",
    ]
    
    for path in possible_paths:
        if path.exists() and path.stat().st_size > 1024:
            # 验证是真实音频文件
            try:
                with open(path, 'rb') as f:
                    header = f.read(12)
                    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                        return str(path)
            except:
                continue
    
    return None

def measure_module_init(config: Dict[str, Any]) -> Dict[str, float]:
    """测量各模块初始化时间"""
    print("\n" + "="*60)
    print("测试1: 模块初始化时间")
    print("="*60)
    
    times = {}
    
    # ASR 模块
    print("\n初始化 ASR 模块...")
    start = time.time()
    from src.asr import FunASRModule
    asr = FunASRModule(config.get('asr', {}))
    times['asr_init'] = time.time() - start
    print(f"  ✓ ASR 初始化: {times['asr_init']:.2f} 秒")
    
    # LLM 模块
    print("\n初始化 LLM 模块...")
    start = time.time()
    from src.llm import LLMInterface
    llm = LLMInterface(config.get('llm', {}))
    times['llm_init'] = time.time() - start
    print(f"  ✓ LLM 初始化: {times['llm_init']:.2f} 秒")
    
    # TTS 模块
    print("\n初始化 TTS 模块...")
    start = time.time()
    from src.tts import IndexTTSModule
    tts = IndexTTSModule(config.get('tts', {}))
    times['tts_init'] = time.time() - start
    print(f"  ✓ TTS 初始化: {times['tts_init']:.2f} 秒")
    
    total_init = times['asr_init'] + times['llm_init'] + times['tts_init']
    times['total_init'] = total_init
    print(f"\n  总初始化时间: {total_init:.2f} 秒")
    
    return times, asr, llm, tts

def measure_individual_modules(
    asr, llm, tts, 
    test_audio: np.ndarray,
    test_text: str,
    reference_audio_path: str,
    num_runs: int = 5
) -> Dict[str, List[float]]:
    """测量各模块单次处理时间"""
    print("\n" + "="*60)
    print("测试2: 各模块单次处理时间 (运行 {} 次取平均值)".format(num_runs))
    print("="*60)
    
    results = {
        'asr': [],
        'llm': [],
        'tts': [],
    }
    
    # ASR 测试
    print("\n测试 ASR 模块...")
    for i in range(num_runs):
        start = time.time()
        asr_text = asr.transcribe_array(test_audio, 16000)
        elapsed = time.time() - start
        results['asr'].append(elapsed)
        print(f"  运行 {i+1}/{num_runs}: {elapsed:.3f} 秒 -> '{asr_text}'")
    print(f"  平均: {mean(results['asr']):.3f} 秒 (±{stdev(results['asr']):.3f})")
    
    # LLM 测试
    print("\n测试 LLM 模块...")
    for i in range(num_runs):
        start = time.time()
        response = llm.chat(test_text, use_history=False)  # 使用 chat 方法，不使用历史记录
        elapsed = time.time() - start
        results['llm'].append(elapsed)
        response_preview = response[:50] if response else "(无响应)"
        print(f"  运行 {i+1}/{num_runs}: {elapsed:.3f} 秒 -> '{response_preview}...'")
    print(f"  平均: {mean(results['llm']):.3f} 秒 (±{stdev(results['llm']):.3f})")
    
    # TTS 测试
    print("\n测试 TTS 模块...")
    for i in range(num_runs):
        start = time.time()
        audio = tts.synthesize(test_text, reference_audio_path=reference_audio_path)
        elapsed = time.time() - start
        results['tts'].append(elapsed)
        print(f"  运行 {i+1}/{num_runs}: {elapsed:.3f} 秒 (生成 {len(audio)} 样本)")
    print(f"  平均: {mean(results['tts']):.3f} 秒 (±{stdev(results['tts']):.3f})")
    
    return results

def measure_end_to_end(
    config: Dict[str, Any],
    test_audio: np.ndarray,
    reference_audio_path: str,
    num_runs: int = 3,
    warmup_runs: int = 1
) -> Dict[str, Any]:
    """测量端到端流程时间"""
    print("\n" + "="*60)
    print("测试3: 端到端流程时间")
    print("="*60)
    print(f"预热运行: {warmup_runs} 次")
    print(f"正式测试: {num_runs} 次")
    
    # 确保配置中有默认参考音频（使用绝对路径）
    if 'tts' not in config:
        config['tts'] = {}
    
    # 转换为绝对路径
    from pathlib import Path as _Path
    _project_root = _Path(__file__).parent.parent
    if not os.path.isabs(reference_audio_path):
        ref_abs = str(_project_root / reference_audio_path)
    else:
        ref_abs = reference_audio_path
    
    if os.path.exists(ref_abs):
        config['tts']['default_reference_audio'] = ref_abs
    
    # 预热运行（不记录时间，包含初始化）
    print("\n预热运行（包含初始化）...")
    from src.pipeline import ConversationPipeline
    import torch
    
    # 清理GPU显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_warmup_init = time.time()
    pipeline_warmup = ConversationPipeline(config)
    warmup_init_time = time.time() - start_warmup_init
    print(f"  预热初始化时间: {warmup_init_time:.2f} 秒")
    
    for i in range(warmup_runs):
        _ = pipeline_warmup.process_audio_array(test_audio, sample_rate=16000)
        print(f"  预热运行 {i+1}/{warmup_runs} 完成")
    
    # 正式测试（复用同一个pipeline，只测量处理时间）
    print("\n正式测试（复用pipeline，只测量处理时间）...")
    process_times = []
    detailed_results = []
    
    for i in range(num_runs):
        print(f"\n端到端运行 {i+1}/{num_runs}...")
        
        # 只测量处理时间（pipeline已初始化）
        start_process = time.time()
        result = pipeline_warmup.process_audio_array(test_audio, sample_rate=16000)
        process_time = time.time() - start_process
        
        process_times.append(process_time)
        
        detail = {
            'run': i + 1,
            'process_time': process_time,
            'success': result.get('success', False),
            'asr_text': result.get('asr_text'),
            'llm_response_length': len(result.get('llm_response', '')) if result.get('llm_response') else 0,
        }
        detailed_results.append(detail)
        
        print(f"  处理时间: {process_time:.3f} 秒")
        if result.get('success'):
            print(f"  ASR: '{result.get('asr_text')}'")
            print(f"  LLM: {result.get('llm_response', '')[:50]}...")
        else:
            print(f"  ✗ 失败: {result.get('error')}")
    
    # 如果需要测量包含初始化的总时间，需要单独测试一次
    # 但先需要完全清理预热pipeline占用的显存
    print("\n单独测试：包含初始化的总时间...")
    print("  正在清理预热pipeline...")
    
    # 删除预热pipeline及其所有组件
    del pipeline_warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()  # 再次清理确保释放
    
    # 等待一下确保显存释放
    time.sleep(1)
    
    start_total_init = time.time()
    try:
        pipeline_init_test = ConversationPipeline(config)
        init_time = time.time() - start_total_init
        
        start_process_test = time.time()
        result_init_test = pipeline_init_test.process_audio_array(test_audio, sample_rate=16000)
        process_time_init_test = time.time() - start_process_test
        
        total_time_with_init = time.time() - start_total_init
        
        # 清理测试用的pipeline
        del pipeline_init_test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        times = [total_time_with_init]  # 只有一次初始化+处理的测量
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ⚠ 显存不足，无法测量包含初始化的总时间: {e}")
        print("  将使用预热阶段的初始化时间作为参考")
        # 使用预热时的初始化时间
        init_time = warmup_init_time
        process_time_init_test = mean(process_times) if process_times else 0
        total_time_with_init = init_time + process_time_init_test
        times = [total_time_with_init]
    
    # 对于包含初始化的时间，只有一次测量
    avg_time = times[0] if times else 0
    std_time = 0  # 只有一次测量，标准差为0
    
    avg_process = mean(process_times)
    std_process = stdev(process_times) if len(process_times) > 1 else 0
    
    print(f"\n  端到端时间 (含初始化): {avg_time:.3f} 秒 (初始化: {init_time:.3f} 秒 + 处理: {process_time_init_test:.3f} 秒)")
    print(f"  平均处理时间 (不含初始化): {avg_process:.3f} 秒 (±{std_process:.3f})")
    print(f"  最快处理时间: {min(process_times):.3f} 秒")
    print(f"  最慢处理时间: {max(process_times):.3f} 秒")
    
    return {
        'times': times,
        'process_times': process_times,
        'average': avg_time,
        'std': std_time,
        'init_time': init_time,
        'average_process': avg_process,
        'std_process': std_process,
        'detailed': detailed_results
    }

def generate_report(
    init_times: Dict[str, float],
    module_times: Dict[str, List[float]],
    e2e_results: Dict[str, Any],
    output_file: str = "benchmark_report.json"
):
    """生成性能报告"""
    print("\n" + "="*60)
    print("性能报告汇总")
    print("="*60)
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'initialization': {
            'asr': init_times['asr_init'],
            'llm': init_times['llm_init'],
            'tts': init_times['tts_init'],
            'total': init_times['total_init'],
        },
        'module_processing': {
            'asr': {
                'mean': mean(module_times['asr']),
                'std': stdev(module_times['asr']) if len(module_times['asr']) > 1 else 0,
                'runs': module_times['asr']
            },
            'llm': {
                'mean': mean(module_times['llm']),
                'std': stdev(module_times['llm']) if len(module_times['llm']) > 1 else 0,
                'runs': module_times['llm']
            },
            'tts': {
                'mean': mean(module_times['tts']),
                'std': stdev(module_times['tts']) if len(module_times['tts']) > 1 else 0,
                'runs': module_times['tts']
            }
        },
        'end_to_end': {
            'mean': e2e_results['average'],
            'std': e2e_results['std'],
            'init_time': e2e_results.get('init_time', 0),
            'mean_process': e2e_results['average_process'],
            'std_process': e2e_results['std_process'],
            'runs': e2e_results['times'],
            'process_runs': e2e_results['process_times'],
            'detailed': e2e_results['detailed']
        }
    }
    
    # 打印报告
    print("\n1. 模块初始化时间:")
    print(f"   ASR:  {report['initialization']['asr']:.2f} 秒")
    print(f"   LLM:  {report['initialization']['llm']:.2f} 秒")
    print(f"   TTS:  {report['initialization']['tts']:.2f} 秒")
    print(f"   总计: {report['initialization']['total']:.2f} 秒")
    
    print("\n2. 模块处理时间 (平均):")
    print(f"   ASR:  {report['module_processing']['asr']['mean']:.3f} 秒 (±{report['module_processing']['asr']['std']:.3f})")
    print(f"   LLM:  {report['module_processing']['llm']['mean']:.3f} 秒 (±{report['module_processing']['llm']['std']:.3f})")
    print(f"   TTS:  {report['module_processing']['tts']['mean']:.3f} 秒 (±{report['module_processing']['tts']['std']:.3f})")
    
    print("\n3. 端到端流程时间:")
    if report['end_to_end']['init_time'] > 0:
        print(f"   初始化时间: {report['end_to_end']['init_time']:.3f} 秒")
    print(f"   总时间 (含初始化): {report['end_to_end']['mean']:.3f} 秒")
    print(f"   平均处理时间 (不含初始化): {report['end_to_end']['mean_process']:.3f} 秒 (±{report['end_to_end']['std_process']:.3f})")
    if report['end_to_end']['process_runs']:
        print(f"   最快处理时间: {min(report['end_to_end']['process_runs']):.3f} 秒")
        print(f"   最慢处理时间: {max(report['end_to_end']['process_runs']):.3f} 秒")
    
    # 保存报告
    output_path = project_root / "data" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存到: {output_path}")
    
    return report

def main():
    """主函数"""
    print("="*60)
    print("智能学伴系统 - 性能基准测试")
    print("="*60)
    
    # 加载配置
    print("\n加载配置...")
    config = load_config()
    
    # 查找参考音频
    print("\n查找参考音频...")
    reference_audio_path = find_reference_audio()
    if not reference_audio_path:
        print("✗ 错误: 未找到有效的参考音频文件")
        print("  请确保存在以下文件之一:")
        print("    - index-tts/examples/test_voice.wav")
        print("    - data/audio_input/*.wav")
        return
    print(f"  ✓ 使用参考音频: {reference_audio_path}")
    
    # 准备测试音频
    print("\n准备测试数据...")
    sample_rate = 16000
    test_text = "你好，我是智能学伴助手，很高兴为你服务"
    
    # 尝试使用TTS生成测试音频（确保有真实语音内容，使用完整音频）
    print(f"  使用TTS生成测试音频（文本: '{test_text}'）...")
    try:
        from src.tts import IndexTTSModule
        
        # 创建临时TTS模块用于生成测试音频
        tts_temp = IndexTTSModule(config.get('tts', {}))
        
        # 生成完整的测试音频（不截取，确保包含完整句子）
        print("    正在生成完整音频...")
        temp_audio = tts_temp.synthesize(
            test_text,
            reference_audio_path=reference_audio_path if reference_audio_path else None
        )
        
        # 使用完整生成的音频（不截取）
        test_audio = temp_audio
        
        # 清理临时TTS模块（释放显存）
        del tts_temp
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        audio_duration = len(test_audio) / sample_rate
        print(f"  ✓ 使用TTS生成完整音频: {len(test_audio)} 样本 ({audio_duration:.2f}秒)")
        print(f"  测试文本: '{test_text}'")
        print(f"  期望ASR识别: 完整句子（{len(test_text)} 字符）")
    except Exception as e:
        print(f"  ⚠ TTS生成失败，使用参考音频: {e}")
        # 回退方案：使用参考音频（尽量使用完整音频或较长片段）
        if reference_audio_path and os.path.exists(reference_audio_path):
            try:
                ref_audio, ref_sr = sf.read(reference_audio_path, dtype='float32')
                if ref_sr != sample_rate:
                    import librosa
                    ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=sample_rate)
                # 使用完整的参考音频（如果太短，则重复）
                # 估计需要的长度：完整句子大约需要6-8秒
                min_duration = 6.0  # 至少6秒
                min_samples = int(sample_rate * min_duration)
                
                if len(ref_audio) >= min_samples:
                    # 如果参考音频足够长，使用完整音频（避免开头可能是静音，从1/10处开始）
                    start_idx = len(ref_audio) // 10
                    test_audio = ref_audio[start_idx:]
                else:
                    # 如果太短，重复音频直到达到最小长度
                    repeat_times = int(np.ceil(min_samples / len(ref_audio)))
                    test_audio = np.tile(ref_audio, repeat_times)[:min_samples]
                
                # 如果是立体声，转换为单声道
                if len(test_audio.shape) > 1:
                    test_audio = np.mean(test_audio, axis=1)
                
                audio_duration = len(test_audio) / sample_rate
                print(f"  ✓ 从参考音频加载: {len(test_audio)} 样本 ({audio_duration:.2f}秒)")
                print(f"  ⚠ 警告: 参考音频可能不包含测试文本，ASR识别结果可能不准确")
            except Exception as e2:
                print(f"  ⚠ 无法加载参考音频，使用生成的测试音频: {e2}")
                # 最后回退：生成较长正弦波（至少6秒）
                duration = 6.0
                t = np.linspace(0, duration, int(sample_rate * duration))
                test_audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
                print(f"  ⚠ 生成正弦波测试音频: {len(test_audio)} 样本 ({duration}秒, 440Hz)")
                print(f"  ⚠ 警告: 正弦波无法被ASR识别为文本，仅用于性能测试")
        else:
            # 生成较长正弦波测试音频
            duration = 6.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            print(f"  ⚠ 生成正弦波测试音频: {len(test_audio)} 样本 ({duration}秒, 440Hz)")
            print(f"  ⚠ 警告: 正弦波无法被ASR识别为文本，仅用于性能测试")
    
    # 测试1: 模块初始化时间
    init_times, asr, llm, tts = measure_module_init(config)
    
    # 测试2: 各模块处理时间
    module_times = measure_individual_modules(
        asr, llm, tts,
        test_audio, test_text,
        reference_audio_path,
        num_runs=5
    )
    
    # 测试3: 端到端流程时间
    e2e_results = measure_end_to_end(
        config,
        test_audio,
        reference_audio_path,
        num_runs=3,  # 端到端测试较慢，减少运行次数
        warmup_runs=1
    )
    
    # 生成报告
    report = generate_report(init_times, module_times, e2e_results)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

if __name__ == "__main__":
    main()

