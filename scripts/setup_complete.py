#!/usr/bin/env python3
"""
智能学伴系统 - 一键环境配置脚本
LearningFriend - Complete Setup Script

功能：自动完成所有环境配置，包括：
  1. 检查Python环境
  2. 安装系统依赖
  3. 安装Python核心依赖
  4. 安装FunASR
  5. 安装IndexTTS2官方代码
  6. 下载模型文件
  7. 配置文件初始化
  8. 验证安装
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# 颜色输出（跨平台）
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_colored(text, color=Colors.NC):
    """打印彩色文本"""
    print(f"{color}{text}{Colors.NC}")

def run_command(cmd, check=True, capture_output=False, shell=False):
    """运行命令"""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            shell=shell,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        return None

def check_command(cmd):
    """检查命令是否存在"""
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            shell=False
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def prompt_user(question):
    """提示用户输入"""
    response = input(f"{question} (y/N): ").strip().lower()
    return response in ['y', 'yes']

def main():
    """主函数"""
    print("=" * 50)
    print_colored("智能学伴系统 - 一键环境配置脚本", Colors.CYAN)
    print("=" * 50)
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.absolute()
    os.chdir(project_root)
    print_colored(f"项目根目录: {project_root}", Colors.BLUE)
    print()
    
    step = 0
    total_steps = 8
    validation_passed = True
    
    # ========== 1. 检查Python环境 ==========
    step += 1
    print_colored(f"[{step}/{total_steps}] 检查Python环境", Colors.GREEN)
    
    python_version = sys.version_info
    print(f"  Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print_colored("  ✗ Python版本过低，需要 >= 3.8", Colors.RED)
        return 1
    
    print_colored("  ✓ Python版本符合要求", Colors.GREEN)
    
    # 检查pip
    try:
        import pip
        print_colored("  ✓ pip可用", Colors.GREEN)
    except ImportError:
        print_colored("  ✗ pip未安装", Colors.RED)
        return 1
    
    # ========== 2. 安装Python核心依赖 ==========
    step += 1
    print_colored(f"\n[{step}/{total_steps}] 安装Python核心依赖", Colors.GREEN)
    
    print("  升级pip...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], capture_output=True)
    
    print("  安装核心依赖包...")
    core_deps = [
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "modelscope>=1.9.0",
        "openai>=1.0.0",
        "requests>=2.31.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "huggingface-hub>=0.19.0",
        "webrtcvad>=2.0.10",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        "transformers==4.52.1",
        "omegaconf>=2.3.0",
        "safetensors>=0.3.0",
        "sentencepiece>=0.2.1",
        "einops>=0.8.1",
        "accelerate>=1.0.0",
        "addict>=2.4.0",
    ]
    
    for dep in core_deps:
        print(f"    - {dep}")
        run_command([sys.executable, "-m", "pip", "install", dep], capture_output=True)
    
    # 验证torchaudio
    try:
        import torchaudio
        print_colored("  ✓ PyTorch和torchaudio安装成功", Colors.GREEN)
    except ImportError:
        print_colored("  ⚠ torchaudio未正确安装，尝试单独安装...", Colors.YELLOW)
        run_command([sys.executable, "-m", "pip", "install", "torchaudio"], capture_output=True)
    
    # ========== 3. 安装FunASR ==========
    step += 1
    print_colored(f"\n[{step}/{total_steps}] 安装FunASR", Colors.GREEN)
    
    funasr_dir = project_root / "FunASR"
    
    if not funasr_dir.exists():
        print("  FunASR目录不存在")
        if prompt_user("  是否克隆FunASR仓库？(约500MB)"):
            print("  克隆FunASR仓库...")
            result = run_command(
                ["git", "clone", "https://github.com/alibaba-damo-academy/FunASR.git", str(funasr_dir)],
                check=False
            )
            if result and result.returncode == 0:
                print_colored("  ✓ FunASR仓库克隆成功", Colors.GREEN)
            else:
                print_colored("  ✗ FunASR仓库克隆失败", Colors.RED)
                if not prompt_user("  是否继续？"):
                    return 1
        else:
            print_colored("  ⚠ 跳过FunASR克隆", Colors.YELLOW)
    
    if funasr_dir.exists():
        print("  安装FunASR...")
        os.chdir(funasr_dir)
        run_command([sys.executable, "-m", "pip", "install", "-e", "."], capture_output=True)
        os.chdir(project_root)
        print_colored("  ✓ FunASR安装完成", Colors.GREEN)
    else:
        print_colored("  ⚠ FunASR目录不存在，跳过安装", Colors.YELLOW)
    
    # ========== 4. 安装IndexTTS2官方代码 ==========
    step += 1
    print_colored(f"\n[{step}/{total_steps}] 安装IndexTTS2官方代码", Colors.GREEN)
    
    indextts_dir = project_root / "index-tts"
    
    if not indextts_dir.exists():
        print("  IndexTTS2目录不存在")
        if prompt_user("  是否克隆IndexTTS2官方仓库？"):
            print("  克隆IndexTTS2官方仓库...")
            result = run_command(
                ["git", "clone", "https://github.com/index-tts/index-tts.git", str(indextts_dir)],
                check=False
            )
            if result and result.returncode == 0:
                print_colored("  ✓ IndexTTS2仓库克隆成功", Colors.GREEN)
                
                # 安装IndexTTS2依赖
                req_file = indextts_dir / "requirements.txt"
                if req_file.exists():
                    print("  安装IndexTTS2依赖...")
                    run_command(
                        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                        capture_output=True
                    )
            else:
                print_colored("  ✗ IndexTTS2仓库克隆失败", Colors.RED)
                if not prompt_user("  是否继续？"):
                    return 1
        else:
            print_colored("  ⚠ 跳过IndexTTS2克隆", Colors.YELLOW)
    else:
        print_colored("  ✓ IndexTTS2目录已存在", Colors.GREEN)
    
    # ========== 5. 配置模型文件 ==========
    step += 1
    print_colored(f"\n[{step}/{total_steps}] 配置模型文件", Colors.GREEN)
    
    # 创建必要的目录
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "funasr").mkdir(exist_ok=True)
    (models_dir / "indextts2").mkdir(exist_ok=True)
    
    data_dir = project_root / "data"
    (data_dir / "audio_input").mkdir(parents=True, exist_ok=True)
    (data_dir / "audio_output").mkdir(parents=True, exist_ok=True)
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    print("  FunASR模型会在首次运行时自动下载")
    print("  IndexTTS2模型需要手动下载...")
    
    # 检查IndexTTS2模型
    indextts_checkpoints = indextts_dir / "checkpoints"
    models_indextts2 = models_dir / "indextts2"
    
    if indextts_checkpoints.exists() and (indextts_checkpoints / "config.yaml").exists():
        print_colored("  ✓ 检测到IndexTTS2官方模型文件", Colors.GREEN)
        
        if not (models_indextts2 / "config.yaml").exists():
            if prompt_user("  是否将IndexTTS2模型复制到models/indextts2？"):
                print("  复制模型文件...")
                try:
                    shutil.copytree(indextts_checkpoints, models_indextts2, dirs_exist_ok=True)
                    print_colored("  ✓ 模型文件复制完成", Colors.GREEN)
                except Exception as e:
                    print_colored(f"  ✗ 复制失败: {e}", Colors.RED)
    else:
        print_colored("  ⚠ 未找到IndexTTS2模型文件", Colors.YELLOW)
        
        if prompt_user("  是否使用Python脚本下载IndexTTS2模型？(约5.9GB)"):
            print("  运行下载脚本...")
            download_script = project_root / "scripts" / "download_indextts2_manual.py"
            if download_script.exists():
                result = run_command(
                    [sys.executable, str(download_script)],
                    check=False
                )
                if not result or result.returncode != 0:
                    print_colored("  ⚠ 模型下载失败，稍后可重试", Colors.YELLOW)
            else:
                print_colored("  ✗ 未找到下载脚本", Colors.RED)
    
    # ========== 6. 配置文件初始化 ==========
    step += 1
    print_colored(f"\n[{step}/{total_steps}] 初始化配置文件", Colors.GREEN)
    
    config_file = project_root / "config" / "config.yaml"
    config_example = project_root / "config" / "config.yaml.example"
    
    if not config_file.exists() and config_example.exists():
        print("  从示例文件创建配置文件...")
        shutil.copy(config_example, config_file)
        print_colored(f"  ✓ 配置文件已创建: {config_file.relative_to(project_root)}", Colors.GREEN)
        print_colored("  ⚠ 请编辑配置文件，填入你的API Key", Colors.YELLOW)
    else:
        if config_file.exists():
            print_colored(f"  ✓ 配置文件已存在: {config_file.relative_to(project_root)}", Colors.GREEN)
        else:
            print_colored("  ⚠ 配置文件不存在，请手动创建", Colors.YELLOW)
    
    # ========== 7. 验证安装 ==========
    step += 1
    print_colored(f"\n[{step}/{total_steps}] 验证安装", Colors.GREEN)
    
    print("  测试模块导入...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print_colored("  ✗ PyTorch导入失败", Colors.RED)
        validation_passed = False
    
    try:
        import torchaudio
        print(f"  ✓ torchaudio {torchaudio.__version__}")
    except ImportError:
        print_colored("  ⚠ torchaudio导入失败（可能不影响使用）", Colors.YELLOW)
    
    try:
        import librosa
        print(f"  ✓ librosa {librosa.__version__}")
    except ImportError:
        print_colored("  ✗ librosa导入失败", Colors.RED)
        validation_passed = False
    
    try:
        import soundfile
        print(f"  ✓ soundfile {soundfile.__version__}")
    except ImportError:
        print_colored("  ✗ soundfile导入失败", Colors.RED)
        validation_passed = False
    
    if funasr_dir.exists():
        try:
            import funasr
            print("  ✓ FunASR安装成功")
        except ImportError:
            print_colored("  ⚠ FunASR导入失败", Colors.YELLOW)
    else:
        print_colored("  ⚠ FunASR目录不存在，跳过验证", Colors.YELLOW)
    
    try:
        from huggingface_hub import snapshot_download
        print("  ✓ huggingface-hub可用")
    except ImportError:
        print_colored("  ⚠ huggingface-hub不可用", Colors.YELLOW)
    
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        print("  ✓ modelscope可用")
    except ImportError:
        print_colored("  ⚠ modelscope不可用", Colors.YELLOW)
    
    # ========== 完成提示 ==========
    print("\n" + "=" * 50)
    if validation_passed:
        print_colored("环境配置完成！", Colors.GREEN)
    else:
        print_colored("环境配置基本完成，但部分验证失败", Colors.YELLOW)
    print("=" * 50)
    print()
    
    print_colored("下一步操作：", Colors.CYAN)
    print("1. 编辑配置文件，填入API Key:")
    print_colored(f"   {config_file}", Colors.BLUE)
    print()
    print("2. 运行端到端测试:")
    print_colored("   python test_pipeline.py", Colors.BLUE)
    print()
    print("3. 开始使用系统:")
    print_colored("   python main.py --mode interactive", Colors.BLUE)
    print()
    print_colored("重要提示：", Colors.YELLOW)
    print("• FunASR模型会在首次运行时自动从ModelScope下载")
    print("• 确保配置文件中的API Key已正确填写")
    print("• 如果IndexTTS2模型未下载，会影响TTS功能")
    print("• 查看 README.md 了解详细使用说明")
    print()
    
    return 0 if validation_passed else 1

if __name__ == "__main__":
    sys.exit(main())

