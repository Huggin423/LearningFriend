#!/usr/bin/env python3
"""
设置 IndexTTS2 官方代码仓库
自动克隆和验证
"""

import os
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
official_repo_path = project_root / "index-tts"

def check_git():
    """检查 git 是否安装"""
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def clone_repo():
    """克隆官方仓库"""
    print("=" * 60)
    print("IndexTTS2 官方代码仓库设置")
    print("=" * 60)
    print()
    
    # 检查 git
    if not check_git():
        print("✗ Git 未安装，无法克隆仓库")
        print("请先安装 Git:")
        print("  Ubuntu/Debian: sudo apt-get install git")
        print("  macOS: brew install git")
        return False
    
    # 如果目录已存在
    if official_repo_path.exists():
        print(f"目录已存在: {official_repo_path}")
        
        # 检查是否是 git 仓库
        if (official_repo_path / '.git').exists():
            print("检测到 .git 目录，尝试更新...")
            try:
                subprocess.run(
                    ['git', '-C', str(official_repo_path), 'pull'],
                    check=True,
                    capture_output=True
                )
                print("✓ 代码已更新")
            except subprocess.CalledProcessError as e:
                print(f"⚠ 更新失败: {e}")
                response = input("是否删除并重新克隆? (y/N): ")
                if response.lower() == 'y':
                    import shutil
                    shutil.rmtree(official_repo_path)
                    print("已删除旧目录")
                else:
                    print("使用现有代码")
        else:
            print("目录存在但不是 git 仓库")
            response = input("是否删除并重新克隆? (y/N): ")
            if response.lower() == 'y':
                import shutil
                shutil.rmtree(official_repo_path)
                print("已删除旧目录")
            else:
                return False
    
    # 克隆仓库
    if not official_repo_path.exists():
        print(f"\n正在克隆仓库到: {official_repo_path}")
        print("这可能需要一些时间...")
        
        try:
            subprocess.run([
                'git', 'clone',
                'https://github.com/index-tts/index-tts.git',
                str(official_repo_path)
            ], check=True)
            print("✓ 仓库克隆成功")
        except subprocess.CalledProcessError as e:
            print(f"✗ 克隆失败: {e}")
            return False
    
    # 验证关键文件
    print("\n验证仓库内容...")
    possible_files = [
        'inference.py',
        'index_tts/inference.py',
        'src/inference.py',
        'README.md',
        'requirements.txt',
    ]
    
    found_files = []
    for file_name in possible_files:
        file_path = official_repo_path / file_name
        if file_path.exists():
            found_files.append(file_name)
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} (未找到)")
    
    if not found_files:
        print("\n⚠ 警告: 未找到任何预期文件")
        print(f"仓库路径: {official_repo_path}")
        if official_repo_path.exists():
            print("\n实际文件/目录:")
            for item in list(official_repo_path.iterdir())[:10]:
                print(f"  - {item.name}")
        return False
    
    # 检查 requirements.txt
    requirements_file = official_repo_path / 'requirements.txt'
    if requirements_file.exists():
        print(f"\n找到 requirements.txt，是否安装依赖? (y/N): ", end='')
        response = input()
        if response.lower() == 'y':
            print("安装依赖...")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install',
                    '-r', str(requirements_file)
                ], check=True)
                print("✓ 依赖安装成功")
            except subprocess.CalledProcessError as e:
                print(f"⚠ 依赖安装失败: {e}")
    
    print("\n" + "=" * 60)
    print("设置完成！")
    print(f"仓库路径: {official_repo_path}")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = clone_repo()
    sys.exit(0 if success else 1)

