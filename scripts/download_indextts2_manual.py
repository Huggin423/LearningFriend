#!/usr/bin/env python3
"""
手动下载 IndexTTS2 官方模型
支持断点续传和进度显示
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_with_huggingface_hub():
    """使用 huggingface-hub 下载"""
    try:
        from huggingface_hub import snapshot_download
        
        repo_id = "IndexTeam/IndexTTS-2"
        local_dir = project_root / "checkpoints"
        
        print(f"开始从 {repo_id} 下载模型...")
        print(f"保存路径: {local_dir.absolute()}")
        print("这可能需要一些时间，请耐心等待...\n")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True  # 支持断点续传
        )
        
        print("\n✓ 模型下载完成！")
        print(f"模型已保存到: {local_dir.absolute()}")
        return True
        
    except ImportError:
        print("✗ huggingface-hub 未安装")
        print("请运行: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"✗ 下载失败: {str(e)}")
        print("\n提示:")
        print("1. 检查网络连接")
        print("2. 尝试使用镜像站点（见文档 docs/MANUAL_DOWNLOAD_TTS.md）")
        return False


def download_with_modelscope():
    """使用 ModelScope（国内镜像）下载"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        import shutil
        
        model_id = "IndexTeam/IndexTTS-2"
        cache_dir = project_root / "checkpoints"
        target_dir = cache_dir
        
        print(f"开始从 ModelScope 下载模型: {model_id}...")
        print(f"保存路径: {target_dir.absolute()}")
        print("使用国内镜像，下载速度可能更快...\n")
        
        # ModelScope 会在 cache_dir 下创建 model_id 子目录
        downloaded_dir = snapshot_download(model_id, cache_dir=str(cache_dir))
        
        # 如果下载到了子目录，需要移动到目标目录
        downloaded_path = Path(downloaded_dir)
        if downloaded_path.name == "IndexTTS-2" and downloaded_path.parent.name == "IndexTeam":
            print(f"\n检测到文件下载到: {downloaded_path.absolute()}")
            print("正在移动文件到目标目录...")
            
            # 确保目标目录存在
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 移动所有文件
            moved_count = 0
            for item in downloaded_path.iterdir():
                target_item = target_dir / item.name
                if target_item.exists():
                    if target_item.is_dir():
                        shutil.rmtree(target_item)
                    else:
                        target_item.unlink()
                
                if item.is_dir():
                    shutil.move(str(item), str(target_item))
                else:
                    shutil.move(str(item), str(target_item))
                moved_count += 1
            
            # 尝试删除空的子目录
            try:
                if downloaded_path.parent.exists() and not any(downloaded_path.parent.iterdir()):
                    downloaded_path.parent.rmdir()
                if downloaded_path.exists():
                    downloaded_path.rmdir()
            except:
                pass
            
            print(f"✓ 已移动 {moved_count} 个文件/目录到目标位置")
        
        print("\n✓ 模型下载完成！")
        print(f"模型已保存到: {target_dir.absolute()}")
        return True
        
    except ImportError:
        print("✗ modelscope 未安装")
        print("请运行: pip install modelscope")
        return False
    except Exception as e:
        print(f"✗ 下载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verify_download():
    """验证下载的文件"""
    checkpoints_dir = project_root / "checkpoints"
    
    # 检查可能的路径（ModelScope 可能下载到子目录）
    possible_paths = [
        checkpoints_dir,
        checkpoints_dir / "IndexTeam" / "IndexTTS-2",
    ]
    
    required_files = [
        'config.yaml',
        'bpe.model',
        'feat1.pt',
        'feat2.pt',
        'qwen0.6bemo4-merge/model-00001-of-00002.safetensors',
        'qwen0.6bemo4-merge/model-00002-of-00002.safetensors',
    ]
    
    print("\n正在验证下载的文件...")
    missing_files = []
    actual_dir = None
    
    # 先找到实际的文件目录
    for possible_path in possible_paths:
        if (possible_path / 'config.yaml').exists():
            actual_dir = possible_path
            print(f"找到模型文件在: {actual_dir.relative_to(project_root)}")
            break
    
    if actual_dir is None:
        print("✗ 未找到模型文件")
        print("\n可能的位置:")
        for path in possible_paths:
            if path.exists():
                print(f"  - {path}")
        return False
    
    # 验证必需文件
    for file in required_files:
        file_path = actual_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {file} ({size:.1f} MB)")
        else:
            print(f"  ✗ {file} (缺失)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠ 缺少 {len(missing_files)} 个文件")
        print(f"实际路径: {actual_dir.absolute()}")
        print("提示: 如果文件在子目录中，可能需要手动移动或重新下载")
        return False
    else:
        print("\n✓ 所有文件验证通过！")
        if actual_dir != checkpoints_dir:
            print(f"\n注意: 文件在 {actual_dir.relative_to(project_root)} 目录下")
            print("如果需要，可以将文件移动到 checkpoints/ 目录")
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("IndexTTS2 官方模型手动下载工具")
    print("=" * 60)
    print()
    
    # 检查是否有可用的下载工具
    has_hf_hub = False
    has_modelscope = False
    
    try:
        import huggingface_hub
        has_hf_hub = True
    except ImportError:
        pass
    
    try:
        import modelscope
        has_modelscope = True
    except ImportError:
        pass
    
    if not has_hf_hub and not has_modelscope:
        print("错误：未安装任何下载工具")
        print("\n请选择安装其中一个：")
        print("  1. pip install huggingface-hub  # 国际用户推荐")
        print("  2. pip install modelscope        # 国内用户推荐")
        return 1
    
    # 选择下载方法
    print("可用下载方法：")
    if has_hf_hub:
        print("  1. HuggingFace Hub (推荐)")
    if has_modelscope:
        print("  2. ModelScope (国内镜像，速度更快)")
    
    # 默认使用第一个可用方法
    if has_modelscope:
        print("\n使用 ModelScope 下载...")
        success = download_with_modelscope()
    elif has_hf_hub:
        print("\n使用 HuggingFace Hub 下载...")
        success = download_with_huggingface_hub()
    else:
        success = False
    
    if success:
        verify_download()
        print("\n" + "=" * 60)
        print("下载完成！现在可以运行测试：")
        print("  python test_pipeline.py")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("下载失败，请查看上面的错误信息")
        print("更多帮助请查看: docs/MANUAL_DOWNLOAD_TTS.md")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

