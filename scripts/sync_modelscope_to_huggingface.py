#!/usr/bin/env python3
"""
将 ModelScope 缓存的模型同步到 HuggingFace 缓存目录
用于解决模型已下载但程序无法识别的问题

创建符合 HuggingFace 标准缓存结构的目录：
models--xxx/
  snapshots/
    [hash]/  -> 指向 ModelScope 缓存（符号链接或复制）
  refs/
    main -> hash
"""

import os
import shutil
import json
import hashlib
from pathlib import Path

def get_snapshot_hash(ms_path):
    """生成 snapshot hash（使用目录路径的哈希）"""
    path_str = str(ms_path.resolve())
    return hashlib.md5(path_str.encode()).hexdigest()[:8]

def create_hf_structure(hf_model_dir, ms_path, model_name):
    """创建符合 HuggingFace 标准的缓存结构"""
    # 创建 snapshots 目录
    snapshots_dir = hf_model_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成 snapshot hash
    snapshot_hash = get_snapshot_hash(ms_path)
    snapshot_path = snapshots_dir / snapshot_hash
    
    # 如果 snapshot 目录已存在且有效，跳过
    if snapshot_path.exists() and any(snapshot_path.iterdir()):
        print(f"    snapshot 目录已存在: {snapshot_path}")
    else:
        # 创建 snapshot 目录的符号链接指向 ModelScope 缓存
        try:
            ms_path_abs = ms_path.resolve()
            snapshot_path.symlink_to(ms_path_abs)
            print(f"    ✓ 已创建 snapshot 符号链接: {snapshot_path} -> {ms_path_abs}")
        except (OSError, PermissionError) as e:
            # 如果符号链接失败，复制文件
            print(f"    ⚠ 符号链接失败 ({e})，改为复制文件...")
            if snapshot_path.is_symlink():
                snapshot_path.unlink()
            elif snapshot_path.exists():
                shutil.rmtree(snapshot_path)
            shutil.copytree(ms_path, snapshot_path, dirs_exist_ok=True)
            print(f"    ✓ 已复制到 snapshot 目录: {snapshot_path}")
    
    # 创建 refs/main 文件，指向 snapshot hash
    refs_dir = hf_model_dir / "refs"
    refs_dir.mkdir(exist_ok=True)
    main_ref = refs_dir / "main"
    main_ref.write_text(snapshot_hash)
    print(f"    ✓ 已创建 refs/main -> {snapshot_hash}")

def sync_model(ms_path, hf_path, model_name):
    """同步单个模型"""
    if not ms_path.exists():
        print(f"  ⚠ ModelScope 缓存不存在: {ms_path}")
        return False
    
    try:
        # 确保目标目录存在
        hf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果目录已存在，检查是否需要更新
        if hf_path.exists():
            snapshots_dir = hf_path / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                print(f"  ✓ HuggingFace 缓存已存在且有效: {hf_path}")
                return True
            else:
                print(f"  ⚠ 缓存目录存在但结构不完整，正在修复...")
                if hf_path.is_symlink():
                    hf_path.unlink()
                elif hf_path.exists():
                    # 只删除内部结构，保留目录
                    for item in hf_path.iterdir():
                        if item.name not in ['.mdl', '.msc', '.mv', 'README.md', 'config.json', 
                                            'configuration.json', 'preprocessor_config.json',
                                            'conformer_shaw.pt', 'model.safetensors']:
                            try:
                                if item.is_dir():
                                    shutil.rmtree(item)
                                else:
                                    item.unlink()
                            except:
                                pass
        
        # 如果不存在或需要重建，创建目录结构
        if not hf_path.exists() or hf_path.is_symlink():
            if hf_path.is_symlink():
                hf_path.unlink()
            
            # 创建主目录（如果不是符号链接）
            if not hf_path.exists():
                hf_path.mkdir(parents=True, exist_ok=True)
            
            # 创建 HuggingFace 标准结构
            print(f"  正在创建 HuggingFace 标准缓存结构...")
            create_hf_structure(hf_path, ms_path, model_name)
            
            return True
    except Exception as e:
        print(f"  ✗ 同步失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("同步 ModelScope 模型到 HuggingFace 缓存目录")
    print("创建符合 HuggingFace 标准结构的缓存")
    print("=" * 60)
    print()
    
    # 缓存目录
    ms_cache_dir = Path(os.environ.get('MODELSCOPE_CACHE', os.path.expanduser('~/.cache/modelscope')))
    hf_cache_dir = Path(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))
    
    print(f"ModelScope 缓存目录: {ms_cache_dir}")
    print(f"HuggingFace 缓存目录: {hf_cache_dir}")
    print()
    
    # 需要同步的模型
    # ModelScope 会将模型名称中的点号转换为三个下划线
    # 例如: w2v-bert-2.0 -> w2v-bert-2___0
    models_to_sync = {
        "facebook/w2v-bert-2.0": ("facebook/w2v-bert-2.0", "models--facebook--w2v-bert-2.0", "facebook/w2v-bert-2___0"),
        "amphion/MaskGCT": ("amphion/MaskGCT", "models--amphion--MaskGCT", None),
        "funasr/campplus": ("funasr/campplus", "models--funasr--campplus", None),
    }
    
    success_count = 0
    for model_name, (ms_path_rel, hf_path_name, ms_alt_path) in models_to_sync.items():
        print(f"处理模型: {model_name}")
        ms_path = ms_cache_dir / "hub" / ms_path_rel
        hf_path = hf_cache_dir / "hub" / hf_path_name
        
        # 如果 ModelScope 路径不存在，尝试备用路径（处理点号转换）
        if not ms_path.exists() and ms_alt_path:
            ms_alt = ms_cache_dir / "hub" / ms_alt_path
            if ms_alt.exists():
                print(f"  使用 ModelScope 转换后的路径: {ms_alt_path}")
                ms_path = ms_alt
        
        if sync_model(ms_path, hf_path, model_name):
            success_count += 1
            
            # 如果 ModelScope 有备用路径（点号转换版本），确保文件同步
            if ms_alt_path:
                ms_alt = ms_cache_dir / "hub" / ms_alt_path
                ms_orig = ms_cache_dir / "hub" / ms_path_rel
                if ms_orig.exists() and ms_alt.exists():
                    # 确保备用目录有必要的文件
                    print(f"  检查并同步 ModelScope 备用路径...")
                    needed_files = ['model.safetensors', 'preprocessor_config.json', 'config.json']
                    for fname in needed_files:
                        orig_file = ms_orig / fname
                        alt_file = ms_alt / fname
                        if orig_file.exists() and not alt_file.exists():
                            try:
                                import shutil
                                shutil.copy2(orig_file, alt_file)
                                print(f"    ✓ 已复制 {fname} 到备用路径")
                            except Exception as e:
                                print(f"    ⚠ 复制 {fname} 失败: {e}")
        print()
    
    print("=" * 60)
    print(f"同步完成！成功: {success_count}/{len(models_to_sync)}")
    print("=" * 60)
    print()
    print("现在程序应该能够正确识别这些模型了")
    print("请重新运行您的程序")

if __name__ == "__main__":
    main()
