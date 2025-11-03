# 国内网络环境配置指南

本指南专门针对中国大陆用户，帮助解决网络访问问题。

## 🌐 镜像站点选项

### 1. ModelScope（推荐）

**优点**：
- ✅ 由阿里云提供，速度最快
- ✅ 专为中国用户优化
- ✅ 已集成到项目配置中

**适用于**：
- IndexTTS2 模型下载
- FunASR 模型下载
- 其他 ModelScope 托管的模型

**配置方法**：

```bash
# ModelScope会自动使用，无需额外配置
# 在配置文件中已默认设置为：
# asr:
#   hub: "ms"  # ModelScope
```

### 2. HuggingFace 镜像

#### HF-Mirror（推荐）

**优点**：
- ✅ 完全镜像 HuggingFace
- ✅ 速度快，稳定性好

**配置方法**：

```bash
# 临时使用（单次命令）
export HF_ENDPOINT=https://hf-mirror.com

# 永久配置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

**使用示例**：

```bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
python scripts/download_huggingface_models.sh
```

#### 其他 HuggingFace 镜像

```bash
# 清华大学镜像
export HF_ENDPOINT=https://hf-mirror.com

# 魔塔社区镜像
export HF_ENDPOINT=https://www.modelscope.cn
```

### 3. Python 包镜像

#### 清华大学镜像

```bash
# 临时使用
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 阿里云镜像

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

#### 豆瓣镜像

```bash
pip install -r requirements.txt -i https://pypi.douban.com/simple/
```

---

## 🛠️ 具体问题解决

### 问题1：HuggingFace 连接失败

**症状**：
```
OSError: [Errno 101] Network is unreachable
ConnectionError: Failed to establish connection to huggingface.co
```

**解决方案**：

**方法A：使用ModelScope（推荐）**

```bash
# 使用ModelScope下载HuggingFace模型
python scripts/download_huggingface_models.sh
# 选择选项 2
```

**方法B：使用HF-Mirror**

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载
python scripts/download_indextts2_manual.py
# 或使用huggingface-cli
huggingface-cli download IndexTeam/IndexTTS-2
```

**方法C：配置环境变量**

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
cat >> ~/.bashrc << 'EOF'

# HuggingFace 镜像配置
export HF_ENDPOINT=https://hf-mirror.com
EOF

# 重新加载
source ~/.bashrc
```

### 问题2：下载IndexTTS2模型慢

**症状**：下载速度很慢，经常中断

**解决方案**：

```bash
# 使用ModelScope（国内用户最佳选择）
python scripts/download_indextts2_modelscope.py

# 或使用手动下载脚本并选择ModelScope
python scripts/download_indextts2_manual.py
# 脚本会自动检测并使用ModelScope
```

### 问题3：克隆GitHub仓库失败

**症状**：
```
fatal: unable to access 'https://github.com/...': Connection refused
```

**解决方案**：

**方法A：使用代理**

```bash
# 设置代理（根据你的代理配置）
git config --global http.proxy http://your-proxy:port
git config --global https.proxy http://your-proxy:port
```

**方法B：使用镜像站点**

```bash
# Gitee镜像（部分项目）
git clone https://gitee.com/alibaba/FunASR.git

# 或使用GitHub加速
# https://ghproxy.com/
git clone https://ghproxy.com/https://github.com/index-tts/index-tts.git
```

**方法C：配置SSH**

```bash
# 使用SSH代替HTTPS（通常更稳定）
git clone git@github.com:index-tts/index-tts.git
```

### 问题4：PyTorch安装慢

**症状**：pip 安装 PyTorch 速度慢或失败

**解决方案**：

```bash
# 方法1：使用清华镜像
pip install torch torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方法2：使用PyTorch官方源（推荐）
# CPU版本
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU版本（CUDA 11.8）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 方法3：使用阿里云镜像
pip install torch torchaudio -i https://mirrors.aliyun.com/pypi/simple/
```

### 问题5：AutoDL/云服务器网络问题

**症状**：在AutoDL或国内云服务器上网络不稳定

**解决方案**：

```bash
# AutoDL通常已经配置好镜像，直接使用即可
# 如果仍有问题，尝试：

# 1. 检查网络连接
ping www.baidu.com

# 2. 配置DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 223.5.5.5" | sudo tee -a /etc/resolv.conf

# 3. 使用ModelScope（云服务器推荐）
# 大部分云服务器对ModelScope有优化
python scripts/download_indextts2_modelscope.py
```

---

## 📝 完整配置示例

### 一次性配置脚本

创建文件 `setup_china_network.sh`：

```bash
#!/bin/bash
# 国内网络环境一键配置脚本

echo "配置国内网络镜像..."

# 1. 配置pip镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置HuggingFace镜像
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 3. 配置Git代理（可选，根据你的代理修改）
# git config --global http.proxy http://your-proxy:port
# git config --global https.proxy http://your-proxy:port

# 4. 重新加载配置
source ~/.bashrc

echo "配置完成！"
echo ""
echo "现在可以运行："
echo "  python scripts/setup_complete.py"
```

使用方法：

```bash
chmod +x setup_china_network.sh
./setup_china_network.sh
```

### Python环境变量配置

创建文件 `setup_china_env.py`：

```python
#!/usr/bin/env python3
"""配置国内网络环境"""
import os
import subprocess

print("配置国内网络镜像...")

# 1. 配置pip镜像
subprocess.run([
    "pip", "config", "set", "global.index-url",
    "https://pypi.tuna.tsinghua.edu.cn/simple"
])

# 2. 配置环境变量
env_file = os.path.expanduser("~/.bashrc")
with open(env_file, "a") as f:
    f.write("\n# HuggingFace 镜像配置\n")
    f.write("export HF_ENDPOINT=https://hf-mirror.com\n")

print("配置完成！")
print("请运行: source ~/.bashrc")
```

---

## 🎯 推荐配置组合

### 普通用户（推荐）

```bash
# 1. 配置pip镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置HF镜像
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 3. 使用ModelScope下载模型
python scripts/download_indextts2_modelscope.py
```

### 开发者/高级用户

```bash
# 1. 配置所有镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 2. 配置Git加速（可选）
git config --global http.proxy http://your-proxy:port

# 3. 配置DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# 4. 开始使用
source ~/.bashrc
python scripts/setup_complete.py
```

---

## 🔍 验证配置

### 检查配置是否生效

```bash
# 检查pip镜像
pip config list

# 检查HF镜像
echo $HF_ENDPOINT

# 测试下载速度
time huggingface-cli download --help

# 测试ModelScope
python -c "from modelscope.hub.snapshot_download import snapshot_download; print('ModelScope可用')"
```

---

## 📚 相关资源

### 镜像站点列表

| 服务 | 镜像地址 | 说明 |
|------|---------|------|
| PyPI | https://pypi.tuna.tsinghua.edu.cn/simple | 清华大学 |
| PyPI | https://mirrors.aliyun.com/pypi/simple/ | 阿里云 |
| HuggingFace | https://hf-mirror.com | HF-Mirror |
| ModelScope | https://www.modelscope.cn | 阿里云 |
| Docker Hub | https://docker.mirrors.ustc.edu.cn | 中科大 |

### 官方文档

- [ModelScope 文档](https://www.modelscope.cn/docs)
- [HF-Mirror 文档](https://hf-mirror.com/docs)
- [清华大学镜像站](https://mirrors.tuna.tsinghua.edu.cn/)

---

## ❓ 常见问题

### Q1: 为什么推荐使用ModelScope？

A: ModelScope由阿里云提供，专门为中国用户优化，下载速度最快，稳定性最好。

### Q2: HF-Mirror和HuggingFace有什么区别？

A: HF-Mirror是完全镜像HuggingFace，功能相同，但在中国访问速度更快。

### Q3: 可以同时配置多个镜像吗？

A: 可以。不同镜像适用于不同服务：
- PyPI镜像：用于pip安装
- HuggingFace镜像：用于模型下载
- GitHub镜像：用于代码克隆

### Q4: 配置镜像后仍然很慢怎么办？

A: 尝试以下方案：
1. 检查你的网络连接
2. 尝试不同的镜像站点
3. 使用代理服务
4. 联系网络管理员

### Q5: AutoDL上的网络配置？

A: AutoDL通常已经优化了网络，直接使用即可。如果遇到问题，优先使用ModelScope下载模型。

---

## 🆘 获取帮助

如果以上方法都无法解决问题，请：

1. 查看项目主README的常见问题
2. 查看相关模块的文档
3. 提交GitHub Issue（附上网络环境信息）
4. 联系项目维护者

---

**注意**：网络配置可能因地区、网络运营商而异，建议根据实际情况灵活调整。

