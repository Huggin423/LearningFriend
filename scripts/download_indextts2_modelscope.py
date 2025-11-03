#!/usr/bin/env python3
"""
从 ModelScope 下载 IndexTTS2 官方模型
仅使用 ModelScope API，自动下载模型文件到指定目录
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_with_modelscope():
    """使用 ModelScope 下载模型"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        
        model_id = "IndexTeam/IndexTTS-2"
        target_dir = project_root / "models" / "indextts2"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"开始从 ModelScope 下载模型: {model_id}")
        print(f"保存路径: {target_dir.absolute()}")
        print("使用ModelScope国内镜像，下载速度更快...\n")
        
        # 使用 ModelScope 下载模型
        downloaded_dir = snapshot_download(
            model_id, 
            cache_dir=str(target_dir),
            local_files_only=False  # 允许从网络下载
        )
        
        print(f"\n✓ 模型下载完成！")
        print(f"模型已保存到: {downloaded_dir}")
        
        # 验证下载的文件
        verify_download(Path(downloaded_dir))
        
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


def verify_download(model_dir: Path):
    """验证下载的文件"""
    print("\n正在验证下载的文件...")
    
    # 必需的核心文件
    required_files = [
        'config.yaml',
        'bpe.model',
        'feat1.pt',
        'feat2.pt',
    ]
    
    # Qwen 模型文件
    qwen_variants = [
        'qwen0.6bemo4-merge/model.safetensors',  # ModelScope 单文件版本
        'qwen0.6bemo4-merge/model-00001-of-00002.safetensors',  # HuggingFace 分片版本1
        'qwen0.6bemo4-merge/model-00002-of-00002.safetensors',  # HuggingFace 分片版本2
    ]
    
    missing_files = []
    found_files = []
    
    # 验证必需文件
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {file} ({size:.1f} MB)")
            found_files.append(file)
        else:
            print(f"  ✗ {file} (缺失)")
            missing_files.append(file)
    
    # 验证 Qwen 模型文件
    qwen_dir = model_dir / 'qwen0.6bemo4-merge'
    qwen_found = False
    
    if qwen_dir.exists():
        for variant in qwen_variants:
            file_path = model_dir / variant
            if file_path.exists():
                size = file_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {variant} ({size:.1f} MB)")
                qwen_found = True
                found_files.append(variant)
                break
    
    if not qwen_found:
        print(f"  ✗ qwen0.6bemo4-merge/model*.safetensors (缺失)")
        missing_files.append('qwen model')
    
    if missing_files:
        print(f"\n⚠ 缺少 {len(missing_files)} 个文件")
        return False
    else:
        print("\n✓ 所有文件验证通过！")
        print(f"找到 {len(found_files)} 个模型文件")
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("IndexTTS2 官方模型下载工具 (ModelScope)")
    print("=" * 60)
    print()
    
    # 检查 ModelScope 是否安装
    try:
        import modelscope
        print(f"✓ ModelScope 已安装 (版本: {modelscope.__version__})")
    except ImportError:
        print("✗ ModelScope 未安装")
        print("\n请先安装 ModelScope:")
        print("  pip install modelscope")
        return 1
    
    # 下载模型
    print("\n开始下载模型...")
    success = download_with_modelscope()
    
    if success:
        print("\n" + "=" * 60)
        print("下载完成！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 确保 config/config.yaml 中 model_path 指向正确的路径")
        print("2. 运行测试: python test_pipeline.py")
        return 0
    else:
        print("\n" + "=" * 60)
        print("下载失败，请查看上面的错误信息")
        print("=" * 60)
        print("\n可能的解决方案：")
        print("1. 检查网络连接")
        print("2. 检查 ModelScope 是否正确安装")
        print("3. 手动下载模型到 models/indextts2 目录")
        return 1


if __name__ == "__main__":
    sys.exit(main())
