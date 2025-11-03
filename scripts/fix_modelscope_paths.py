#!/usr/bin/env python3
"""
修复 ModelScope 路径问题
ModelScope 会将模型名称中的点号转换为三个下划线
例如: w2v-bert-2.0 -> w2v-bert-2___0

这个脚本确保两个路径都有必要的模型文件
"""

import os
import shutil
from pathlib import Path

def fix_model_paths():
    """修复 ModelScope 模型路径问题"""
    print("=" * 60)
    print("修复 ModelScope 路径问题")
    print("=" * 60)
    print()
    
    ms_cache_dir = Path(os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope')))
    
    # facebook/w2v-bert-2.0 的路径转换
    orig_path = ms_cache_dir / "hub" / "facebook" / "w2v-bert-2.0"
    alt_path = ms_cache_dir / "hub" / "facebook" / "w2v-bert-2___0"
    
    needed_files = ['model.safetensors', 'preprocessor_config.json', 'config.json']
    
    if orig_path.exists() and alt_path.exists():
        print(f"原始路径: {orig_path}")
        print(f"ModelScope 转换路径: {alt_path}")
        print()
        
        fixed_count = 0
        for fname in needed_files:
            orig_file = orig_path / fname
            alt_file = alt_path / fname
            
            if orig_file.exists() and not alt_file.exists():
                try:
                    shutil.copy2(orig_file, alt_file)
                    print(f"✓ 已复制 {fname} 到转换路径")
                    fixed_count += 1
                except Exception as e:
                    print(f"✗ 复制 {fname} 失败: {e}")
            elif orig_file.exists() and alt_file.exists():
                print(f"✓ {fname} 已存在于转换路径")
            elif not orig_file.exists():
                print(f"⚠ {fname} 在原始路径中不存在")
        
        print()
        if fixed_count > 0:
            print(f"修复完成！已复制 {fixed_count} 个文件")
        else:
            print("所有文件都已存在，无需修复")
    else:
        if not orig_path.exists():
            print(f"⚠ 原始路径不存在: {orig_path}")
        if not alt_path.exists():
            print(f"⚠ ModelScope 转换路径不存在: {alt_path}")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    fix_model_paths()

