# 脚本使用说明

本目录包含智能学伴系统的各种配置和安装脚本。

## 🌐 网络环境

**国内用户提示**：中国大陆访问 HuggingFace 和 GitHub 可能较慢，建议：
- 使用 ModelScope 镜像下载模型（已在配置中默认启用）
- 配置国内镜像源（详见 [NETWORK_GUIDE.md](NETWORK_GUIDE.md)）
- 使用 `download_indextts2_modelscope.py` 下载 IndexTTS2 模型

## 🚀 推荐：一键环境配置

**在任何机器上首次配置环境时，请使用统一的环境配置脚本：**

### Windows 用户

```bash
python scripts/setup_complete.py
```

### Linux/macOS 用户

```bash
# 方式1：直接运行Python脚本
python3 scripts/setup_complete.py

# 方式2：运行Shell脚本（推荐）
bash scripts/setup_complete.sh

# 方式3：如果已添加执行权限
./scripts/setup_complete.sh
```

### 功能说明

这个统一脚本会自动完成以下所有步骤：

1. ✅ 检查Python环境（版本 >= 3.8）
2. ✅ 安装Python核心依赖（PyTorch、FunASR、IndexTTS2等）
3. ✅ 克隆并安装FunASR仓库
4. ✅ 克隆并安装IndexTTS2官方代码
5. ✅ 配置模型文件目录
6. ✅ 初始化配置文件
7. ✅ 验证安装是否成功

### 交互式选项

脚本会询问你：
- 是否克隆FunASR仓库？（约500MB）
- 是否克隆IndexTTS2仓库？
- 是否下载IndexTTS2模型？（约5.9GB）
- 是否将模型文件复制到指定目录？

### 完成后

根据提示完成以下步骤：

1. **编辑配置文件**，填入你的API Key：
   ```bash
   # 复制示例配置
   cp config/config.yaml.example config/config.yaml
   
   # 编辑配置文件
   vim config/config.yaml  # 或使用其他编辑器
   ```

2. **运行测试**：
   ```bash
   python test_pipeline.py
   ```

3. **开始使用**：
   ```bash
   python main.py --mode interactive
   ```

---

## 📁 其他脚本文件说明

### 已整合的脚本

以下功能已经被整合到 `setup_complete.py` 中：

| 原脚本文件 | 功能 | 状态 |
|-----------|------|------|
| `setup_funasr.sh` | 安装FunASR | ✅ 已整合 |
| `setup_indextts2_official.py` | 克隆IndexTTS2 | ✅ 已整合 |
| `install_autodl.sh` | AutoDL环境安装 | ✅ 已整合（通用化） |
| `download_models.sh` | 创建模型目录 | ✅ 已整合 |

### 仍然可用的独立脚本

#### 模型下载相关

##### 1. `download_indextts2_manual.py`

**用途**：手动下载IndexTTS2官方模型

**何时使用**：
- 在一键配置时选择不下载模型
- 后续需要补充下载模型
- 模型文件损坏需要重新下载

**使用**：
```bash
python scripts/download_indextts2_manual.py
```

**特点**：
- 支持从HuggingFace Hub下载（国际用户）
- 支持从ModelScope下载（国内用户，速度更快）
- 自动检测可用下载工具
- 支持断点续传
- 下载后自动验证文件完整性

##### 2. `download_indextts2_modelscope.py`

**用途**：仅使用ModelScope下载IndexTTS2模型

**何时使用**：
- 在国内网络环境
- 只需要ModelScope方式下载

**使用**：
```bash
python scripts/download_indextts2_modelscope.py
```

##### 3. `download_huggingface_models.sh`

**用途**：下载IndexTTS2所需的HuggingFace模型（用于离线环境）

**使用**：
```bash
bash scripts/download_huggingface_models.sh
```

**前提**：
- 需要先安装：`pip install huggingface_hub[cli]`

**下载的模型**：
- `facebook/w2v-bert-2.0`
- `amphion/MaskGCT`
- `funasr/camppplus`

**国内网络配置**：
- ✅ 自动提示选择镜像源（HuggingFace/ModelScope/HF-Mirror）
- ✅ 默认使用ModelScope镜像（推荐国内用户）
- ✅ 支持HF-Mirror国内镜像

#### 迁移和修复相关

##### 4. `migrate_to_indextts_checkpoints.py`

**用途**：将旧版模型文件迁移到新的目录结构

**何时使用**：
- 从旧版本升级
- 模型文件路径混乱
- 需要统一模型文件位置

**使用**：
```bash
python scripts/migrate_to_indextts_checkpoints.py
```

**功能**：
- 查找旧版本模型文件（`checkpoints/` 目录）
- 移动到新的统一目录（`index-tts/checkpoints/`）
- 自动合并重复文件
- 验证文件完整性

##### 5. `fix_modelscope_path.py`

**用途**：修复ModelScope下载路径问题

**何时使用**：
- 模型被下载到 `checkpoints/IndexTeam/IndexTTS-2/` 嵌套目录
- 需要移动到 `checkpoints/` 根目录

**使用**：
```bash
python scripts/fix_modelscope_path.py
```

---

## 🔍 故障排查

### 问题1：FunASR克隆失败

**症状**：`git clone` 命令失败

**解决方案**：
1. 检查网络连接
2. 手动克隆：
   ```bash
   git clone https://github.com/alibaba-damo-academy/FunASR.git
   ```
3. 如果在中国大陆，考虑使用镜像或代理

### 问题2：IndexTTS2模型下载失败

**症状**：下载过程中断或失败

**解决方案**：
```bash
# 方式1：重试下载（支持断点续传）
python scripts/download_indextts2_manual.py

# 方式2：使用ModelScope（国内用户推荐）
python scripts/download_indextts2_modelscope.py

# 方式3：手动下载
# 查看 docs/MANUAL_DOWNLOAD_TTS.md 获取详细说明
```

**国内网络特殊处理**：
如果遇到 `Network is unreachable` 错误，请：
1. 使用ModelScope下载（推荐）
2. 配置HF-Mirror镜像：`export HF_ENDPOINT=https://hf-mirror.com`
3. 查看 [NETWORK_GUIDE.md](NETWORK_GUIDE.md) 获取详细网络配置指南

### 问题3：依赖安装失败

**症状**：`pip install` 报错

**解决方案**：
```bash
# 1. 升级pip
python -m pip install --upgrade pip

# 2. 尝试使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 逐个安装问题包
pip install <package_name> --verbose
```

### 问题4：配置文件问题

**症状**：找不到配置文件或配置错误

**解决方案**：
```bash
# 1. 从示例文件创建
cp config/config.yaml.example config/config.yaml

# 2. 检查配置文件语法
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

### 问题5：Git未安装

**症状**：克隆仓库失败

**解决方案**：

**Windows**：
- 下载并安装 [Git for Windows](https://git-scm.com/download/win)

**Linux (Ubuntu/Debian)**：
```bash
sudo apt-get update
sudo apt-get install git
```

**macOS**：
```bash
# 使用Homebrew
brew install git

# 或下载安装包
# https://git-scm.com/download/mac
```

---

## 📝 各脚本详细说明

### setup_complete.py / setup_complete.sh

**一行命令完成所有配置**的首选脚本。

**特点**：
- ✅ 跨平台（Windows/Linux/macOS）
- ✅ 交互式提示
- ✅ 自动检测已有安装
- ✅ 完整的错误处理
- ✅ 安装验证

**使用场景**：
- 🆕 全新环境配置
- 🔄 在其他机器上配置
- 🔧 重新配置环境

---

### download_indextts2_manual.py

IndexTTS2模型下载工具（智能版）。

**特点**：
- 自动检测可用的下载工具（HuggingFace/ModelScope）
- 支持断点续传
- 自动验证文件完整性
- 详细的进度显示

**模型信息**：
- 模型ID：`IndexTeam/IndexTTS-2`
- 大小：约5.9GB
- 必需文件：
  - `config.yaml`
  - `bpe.model`
  - `feat1.pt`
  - `feat2.pt`
  - `qwen0.6bemo4-merge/`（目录）

---

### download_indextts2_modelscope.py

IndexTTS2模型下载工具（ModelScope专版）。

**特点**：
- 仅使用ModelScope下载
- 适合国内网络环境
- 下载到 `models/indextts2/` 目录

---

### migrate_to_indextts_checkpoints.py

模型文件迁移工具。

**功能**：
- 搜索旧版模型文件位置
- 移动到统一目录
- 合并重复文件
- 清理空目录

**从 → 到**：
```
checkpoints/              → index-tts/checkpoints/
checkpoints/IndexTeam/    → index-tts/checkpoints/
IndexTTS-2/
```

---

### fix_modelscope_path.py

修复ModelScope下载路径嵌套问题。

**问题示例**：
```
checkpoints/
└── IndexTeam/
    └── IndexTTS-2/
        ├── config.yaml
        └── ...
```

**修复后**：
```
checkpoints/
├── config.yaml
└── ...
```

---

## 🎯 快速参考

### 新机器首次配置

```bash
# 1. 一键配置（推荐）
python scripts/setup_complete.py

# 2. 编辑配置文件
cp config/config.yaml.example config/config.yaml
vim config/config.yaml  # 填入API Key

# 3. 测试
python test_pipeline.py

# 4. 开始使用
python main.py --mode interactive
```

### 仅下载模型

```bash
# 方式1：自动选择下载源
python scripts/download_indextts2_manual.py

# 方式2：使用ModelScope（国内）
python scripts/download_indextts2_modelscope.py
```

### 从旧版本升级

```bash
# 1. 迁移模型文件
python scripts/migrate_to_indextts_checkpoints.py

# 2. 更新依赖
pip install -r requirements.txt --upgrade

# 3. 验证
python test_pipeline.py
```

### 修复常见问题

```bash
# 路径嵌套问题
python scripts/fix_modelscope_path.py

# 重新安装FunASR
cd FunASR && pip install -e . && cd ..

# 验证配置
python -c "from config import load_config; load_config()"
```

---

## 💡 最佳实践

1. **首次配置**：使用 `setup_complete.py` 一键完成
2. **更新依赖**：定期运行 `pip install -r requirements.txt --upgrade`
3. **模型管理**：使用统一目录 `models/indextts2/` 和 `models/funasr/`
4. **配置备份**：定期备份你的 `config/config.yaml`
5. **环境隔离**：建议使用虚拟环境（venv 或 conda）

---

## 📚 相关文档

- [项目主README](../README.md)
- [ASR模块文档](../src/asr/README.md)
- [LLM模块文档](../src/llm/README.md)
- [TTS模块文档](../src/tts/README.md)
- [手动下载TTS模型](../docs/MANUAL_DOWNLOAD_TTS.md)
- [AutoDL部署指南](../docs/DEPLOY_AUTODL.md)

---

## 🤝 获取帮助

如果遇到问题：

1. 查看本README的故障排查部分
2. 检查项目主README的常见问题
3. 查看相关模块的文档
4. 提交GitHub Issue

---

**Happy Coding! 🎉**

