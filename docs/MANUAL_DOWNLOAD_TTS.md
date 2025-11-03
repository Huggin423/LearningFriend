# IndexTTS2 官方模型手动下载指南

本指南介绍如何手动下载 IndexTTS2 官方模型，适用于网络不稳定或自动下载失败的情况。

## 📋 模型信息

- **模型仓库**: `IndexTeam/IndexTTS-2`
- **HuggingFace 链接**: https://huggingface.co/IndexTeam/IndexTTS-2
- **模型大小**: 约 5.9GB
- **本地存储路径**: `checkpoints/` (默认)

## 🔧 方法一：使用 huggingface-cli（推荐）

### 1. 安装 huggingface-cli

```bash
pip install huggingface-hub
```

### 2. 登录（可选，公开模型无需登录）

```bash
huggingface-cli login
```

### 3. 下载模型

```bash
# 在项目根目录执行
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints
```

或者指定完整路径：

```bash
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir ~/LearningFriend/checkpoints
```

## 🔧 方法二：使用 Git LFS（适合有 Git 环境）

### 1. 安装 Git LFS

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# 初始化 Git LFS
git lfs install
```

### 2. 克隆模型仓库

```bash
# 在项目根目录执行
cd ~/LearningFriend
git clone https://huggingface.co/IndexTeam/IndexTTS-2 checkpoints
```

如果已存在 checkpoints 目录，可以克隆到临时目录再移动：

```bash
git clone https://huggingface.co/IndexTeam/IndexTTS-2 checkpoints_temp
mv checkpoints_temp/* checkpoints/
rm -rf checkpoints_temp
```

## 🔧 方法三：使用 Python 脚本下载

创建下载脚本 `scripts/download_indextts2_manual.py`:

```python
#!/usr/bin/env python3
"""
手动下载 IndexTTS2 官方模型
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_models():
    """下载 IndexTTS2 官方模型"""
    repo_id = "IndexTeam/IndexTTS-2"
    local_dir = Path("checkpoints")
    
    print(f"开始从 {repo_id} 下载模型...")
    print(f"保存路径: {local_dir.absolute()}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True  # 支持断点续传
        )
        print("✓ 模型下载完成！")
        print(f"模型已保存到: {local_dir.absolute()}")
    except Exception as e:
        print(f"✗ 下载失败: {str(e)}")
        print("\n提示:")
        print("1. 检查网络连接")
        print("2. 尝试使用镜像站点")
        print("3. 使用其他下载方法（见文档）")
        raise

if __name__ == "__main__":
    download_models()
```

运行脚本：

```bash
python scripts/download_indextts2_manual.py
```

## 🔧 方法四：使用镜像站点（国内用户推荐）

### 使用 ModelScope（阿里云）

```bash
# 安装 ModelScope
pip install modelscope

# 下载模型
python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', cache_dir='checkpoints')"
```

或创建脚本：

```python
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download('IndexTeam/IndexTTS-2', cache_dir='checkpoints')
```

## 📁 验证下载

下载完成后，检查以下文件是否都存在：

```bash
ls -lh checkpoints/
```

应该包含：
- `config.yaml` - 配置文件
- `bpe.model` - BPE 分词模型
- `feat1.pt` - 特征文件 1
- `feat2.pt` - 特征文件 2
- `qwen0.6bemo4-merge/model-00001-of-00002.safetensors` - Qwen 模型文件 1
- `qwen0.6bemo4-merge/model-00002-of-00002.safetensors` - Qwen 模型文件 2

## 🚀 使用下载的模型

下载完成后，重新运行测试：

```bash
python test_pipeline.py
```

系统会自动检测 `checkpoints/` 目录中的模型文件并使用它们。

## 🔍 故障排除

### 问题1: 下载速度慢

**解决方案**：
- 使用国内镜像（ModelScope）
- 使用代理
- 分时段下载（避开高峰）

### 问题2: 下载中断

**解决方案**：
```bash
# huggingface-cli 支持断点续传，重新运行相同命令即可
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints
```

### 问题3: 磁盘空间不足

**解决方案**：
```bash
# 检查磁盘空间
df -h

# 清理缓存（如果需要）
rm -rf ~/.cache/huggingface/
```

### 问题4: 权限问题

**解决方案**：
```bash
# 确保有写入权限
chmod -R 755 checkpoints/

# 或使用 sudo（不推荐）
sudo chown -R $USER:$USER checkpoints/
```

## 📝 配置文件设置

确保 `config/config.yaml` 中设置了正确的路径：

```yaml
tts:
  use_official: true
  model_path: "checkpoints"  # 模型目录
  official_repo: "index-tts"  # 官方代码仓库路径
```

## 🔗 相关链接

- [IndexTTS2 GitHub](https://github.com/index-tts/index-tts)
- [HuggingFace 模型页面](https://huggingface.co/IndexTeam/IndexTTS-2)
- [HuggingFace Hub 文档](https://huggingface.co/docs/huggingface_hub)
- [ModelScope 文档](https://modelscope.cn/docs)

## 💡 提示

1. **首次下载**: 建议使用 `huggingface-cli`，速度较快且支持断点续传
2. **国内用户**: 使用 ModelScope 镜像下载速度更快
3. **离线使用**: 下载后可以将 `checkpoints/` 目录备份，后续直接复制即可
4. **版本管理**: 模型文件较大，建议添加到 `.gitignore`，不要提交到 Git

## ✅ 完成检查清单

- [ ] 已安装 `huggingface-hub` 或 `modelscope`
- [ ] 已下载模型到 `checkpoints/` 目录
- [ ] 已验证所有必需文件都存在
- [ ] 已配置正确的 `config.yaml`
- [ ] 已成功运行测试 `python test_pipeline.py`

