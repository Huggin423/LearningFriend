#!/usr/bin/env python3
"""
将模型文件迁移到 index-tts/checkpoints/ 目录
统一管理模型文件
"""

import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent

def migrate_models():
    """迁移模型文件"""
    print("=" * 60)
    print("迁移模型文件到 index-tts/checkpoints/")
    print("=" * 60)
    print()
    
    # 源目录（可能的位置）
    possible_sources = [
        project_root / "checkpoints",
        project_root / "checkpoints" / "IndexTeam" / "IndexTTS-2",
    ]
    
    # 目标目录
    target_dir = project_root / "index-tts" / "checkpoints"
    
    # 确保目标目录存在
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {target_dir.absolute()}")
    
    # 查找源文件
    source_dir = None
    for possible_source in possible_sources:
        if possible_source.exists():
            # 检查是否有模型文件（config.yaml 是标志文件）
            if (possible_source / "config.yaml").exists():
                source_dir = possible_source
                print(f"找到源目录: {source_dir.absolute()}")
                break
    
    if source_dir is None:
        print("未找到需要迁移的模型文件")
        print("可能的位置:")
        for path in possible_sources:
            if path.exists():
                print(f"  - {path}")
        print("\n模型文件可能已经在 index-tts/checkpoints/ 中")
        return True
    
    if source_dir == target_dir:
        print("源目录和目标目录相同，无需迁移")
        return True
    
    print(f"\n从 {source_dir.relative_to(project_root)} 迁移到 {target_dir.relative_to(project_root)}")
    print()
    
    # 需要迁移的文件/目录
    important_files = [
        'config.yaml',
        'bpe.model',
        'feat1.pt',
        'feat2.pt',
        'qwen0.6bemo4-merge',
        'gpt.pth',
        's2mel.pth',
        'model.safetensors',  # ModelScope 版本
        'wav2vec2bert_stats.pt',
    ]
    
    moved_count = 0
    skipped_count = 0
    
    # 遍历源目录中的所有文件
    for item in source_dir.iterdir():
        # 跳过 .git 和隐藏文件
        if item.name.startswith('.'):
            continue
        
        target_item = target_dir / item.name
        
        # 如果目标已存在，检查是否需要覆盖
        if target_item.exists():
            if item.is_file() and item.stat().st_size == target_item.stat().st_size:
                print(f"  ⊘ {item.name} (已存在，跳过)")
                skipped_count += 1
                continue
            elif item.is_dir():
                # 目录已存在，合并内容
                print(f"  → {item.name}/ (合并目录)")
                # 递归复制目录内容
                for subitem in item.rglob('*'):
                    if subitem.is_file():
                        rel_path = subitem.relative_to(item)
                        target_subitem = target_item / rel_path
                        target_subitem.parent.mkdir(parents=True, exist_ok=True)
                        if not target_subitem.exists():
                            shutil.copy2(subitem, target_subitem)
                            moved_count += 1
                continue
        
        # 移动文件或目录
        try:
            if item.is_dir():
                shutil.move(str(item), str(target_item))
                print(f"  ✓ {item.name}/ (目录)")
            else:
                shutil.move(str(item), str(target_item))
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  ✓ {item.name} ({size_mb:.1f} MB)")
            moved_count += 1
        except Exception as e:
            print(f"  ✗ {item.name} (移动失败: {e})")
    
    # 清理空的源目录
    try:
        if source_dir.exists() and not any(source_dir.iterdir()):
            source_dir.rmdir()
            print(f"\n删除空目录: {source_dir.name}")
        
        # 如果是子目录，尝试删除父目录
        if source_dir.parent.name == "IndexTeam" and not any(source_dir.parent.iterdir()):
            source_dir.parent.rmdir()
            print(f"删除空目录: {source_dir.parent.name}")
    except Exception as e:
        print(f"清理目录时出错: {e}")
    
    print(f"\n迁移完成!")
    print(f"  移动: {moved_count} 项")
    print(f"  跳过: {skipped_count} 项")
    print(f"\n目标目录: {target_dir.absolute()}")
    
    # 验证关键文件
    print("\n验证文件...")
    required_files = ['config.yaml']
    missing = []
    for file in required_files:
        if (target_dir / file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (缺失)")
            missing.append(file)
    
    if missing:
        print(f"\n⚠ 缺少关键文件: {missing}")
        return False
    
    return True

if __name__ == "__main__":
    success = migrate_models()
    
    if success:
        print("\n" + "=" * 60)
        print("下一步:")
        print("1. 更新 config/config.yaml，设置 model_path: 'index-tts/checkpoints'")
        print("2. 运行测试: python test_pipeline.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("迁移过程中出现问题，请检查上面的错误信息")
        print("=" * 60)
    
    exit(0 if success else 1)



