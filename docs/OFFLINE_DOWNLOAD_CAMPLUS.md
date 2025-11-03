# 离线环境下载 funasr/campplus 模型

## 问题说明

`funasr/campplus` 模型中的 `campplus_cn_common.bin` 文件是 IndexTTS2 必需的，但该模型在 ModelScope 上不可用，只能从 HuggingFace 下载。

## 解决方案

### 方案1：在有网络的环境中预先下载

如果你有可以访问 HuggingFace 的网络环境（或使用代理/镜像），运行：

```bash
# 使用 HuggingFace 官方站点
huggingface-cli download funasr/campplus

# 或使用 HF-Mirror 镜像（国内推荐）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download funasr/campplus
```

### 方案2：使用 HF-Mirror 镜像（推荐国内用户）

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
huggingface-cli download funasr/campplus --include campplus_cn_common.bin
```

### 方案3：手动下载文件

1. 访问 HuggingFace 模型页面：https://huggingface.co/funasr/campplus
2. 下载 `campplus_cn_common.bin` 文件
3. 放置到缓存目录：
   ```
   ~/.cache/huggingface/hub/models--funasr--campplus/snapshots/[hash]/campplus_cn_common.bin
   ```

## 验证下载

下载完成后，运行以下命令验证：

```bash
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.environ['HF_HUB_OFFLINE'] = '1'
path = hf_hub_download('funasr/campplus', filename='campplus_cn_common.bin', local_files_only=True)
print(f'✓ 文件已找到: {path}')
"
```

## 注意

- `funasr/campplus` 在 ModelScope 上不存在，无法通过 ModelScope 下载
- 必须从 HuggingFace 或镜像站点下载
- 如果网络不可达，需要预先在有网络的环境中下载

