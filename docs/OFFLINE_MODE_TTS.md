# IndexTTS2 离线模式使用指南

## 问题说明

IndexTTS2 在初始化时需要从 HuggingFace 下载一些依赖模型，包括：
- `facebook/w2v-bert-2.0` - 特征提取器
- `amphion/MaskGCT` - 语义编码器
- `funasr/campplus` - 说话人识别模型

如果网络不可达，模型初始化会失败。

## 解决方案

### 方法 1：预先下载模型（推荐）

在有网络的环境中，预先下载所需模型：

```bash
# 安装 huggingface-cli（如果未安装）
pip install huggingface_hub[cli]

# 下载所需模型
bash scripts/download_huggingface_models.sh

# 或者手动下载
huggingface-cli download facebook/w2v-bert-2.0
huggingface-cli download amphion/MaskGCT
huggingface-cli download funasr/campplus
```

下载的模型会保存在 `~/.cache/huggingface/hub/` 目录下。

### 方法 2：使用镜像

如果 HuggingFace 访问受限，可以使用镜像：

```bash
# 设置 HuggingFace 镜像（例如使用 HuggingFace 中国镜像）
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行下载脚本
bash scripts/download_huggingface_models.sh
```

### 方法 3：手动复制缓存

如果其他机器已经有缓存，可以直接复制：

```bash
# 从有网络的机器复制缓存
scp -r user@remote_machine:~/.cache/huggingface/hub/ ~/.cache/huggingface/
```

## 验证缓存

检查模型是否已下载：

```bash
ls ~/.cache/huggingface/hub/models--facebook--w2v-bert-2.0/
ls ~/.cache/huggingface/hub/models--amphion--MaskGCT/
ls ~/.cache/huggingface/hub/models--funasr--campplus/
```

## 离线模式

代码已自动启用离线模式，会优先使用本地缓存。如果缓存中缺少模型，会显示友好的错误提示和解决方案。

## 注意事项

1. **首次使用**：需要网络连接下载依赖模型（约 500MB-1GB）
2. **后续使用**：可以使用离线模式，无需网络
3. **缓存位置**：默认在 `~/.cache/huggingface/hub/`
4. **缓存大小**：所有依赖模型总计约 1-2GB

