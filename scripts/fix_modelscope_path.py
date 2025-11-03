#!/usr/bin/env python3
"""
修复 ModelScope 下载路径问题
将 checkpoints/IndexTeam/IndexTTS-2 中的文件移动到 checkpoints/
"""

import shutil
from pathlib import Path

def fix_path():
    """修复路径"""
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "checkpoints" / "IndexTeam" / "IndexTTS-2"
    target_dir = project_root / "checkpoints"
    
    if not source_dir.exists():
        print(f"源目录不存在: {source_dir}")
        print("文件可能已经正确位置，或需要重新下载")
        return False
    
    print(f"找到模型文件在: {source_dir}")
    print(f"目标位置: {target_dir}")
    print("\n开始移动文件...")
    
    moved_count = 0
    for item in source_dir.iterdir():
        target_item = target_dir / item.name
        
        # 如果目标已存在，先删除
        if target_item.exists():
            if target_item.is_dir():
                shutil.rmtree(target_item)
                print(f"  删除已存在的目录: {item.name}")
            else:
                target_item.unlink()
                print(f"  删除已存在的文件: {item.name}")
        
        # 移动文件或目录
        if item.is_dir():
            shutil.move(str(item), str(target_item))
            print(f"  ✓ 移动目录: {item.name}")
        else:
            shutil.move(str(item), str(target_item))
            print(f"  ✓ 移动文件: {item.name}")
        moved_count += 1
    
    # 尝试删除空的子目录
    try:
        if source_dir.exists() and not any(source_dir.iterdir()):
            source_dir.rmdir()
            print(f"\n✓ 删除空目录: {source_dir.name}")
        
        parent_dir = source_dir.parent
        if parent_dir.exists() and not any(parent_dir.iterdir()):
            parent_dir.rmdir()
            print(f"✓ 删除空目录: {parent_dir.name}")
    except Exception as e:
        print(f"警告: 无法删除空目录: {e}")
    
    print(f"\n✓ 完成！已移动 {moved_count} 个文件/目录")
    print(f"现在文件在: {target_dir}")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("修复 ModelScope 下载路径")
    print("=" * 60)
    print()
    
    if fix_path():
        print("\n" + "=" * 60)
        print("修复完成！现在可以运行测试：")
        print("  python test_pipeline.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("修复失败，请检查路径")
        print("=" * 60)

