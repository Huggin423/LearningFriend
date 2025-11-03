#!/bin/bash
# 激进的磁盘清理脚本
# 将大文件移动到数据盘

set -e

echo "======================================"
echo "激进磁盘清理 - 移动大文件到数据盘"
echo "======================================"
echo ""

# 检查数据盘
DATA_DISK="/root/autodl-tmp"
if [ ! -d "$DATA_DISK" ]; then
    echo "错误: 数据盘 $DATA_DISK 不存在"
    exit 1
fi

echo "当前系统盘使用情况："
df -h / | tail -1
echo ""
echo "数据盘使用情况："
df -h "$DATA_DISK" | tail -1
echo ""

# 创建数据盘目录结构
CACHE_DIR="$DATA_DISK/.cache"
mkdir -p "$CACHE_DIR"
mkdir -p "$CACHE_DIR/modelscope"
mkdir -p "$CACHE_DIR/huggingface"

echo "开始移动大文件..."
echo ""

# 1. 移动 ModelScope 缓存（最大，17G）
if [ -d "/root/.cache/modelscope" ] && [ ! -L "/root/.cache/modelscope" ]; then
    echo "1. 移动 ModelScope 缓存 (约 17G)..."
    SIZE=$(du -sh /root/.cache/modelscope 2>/dev/null | cut -f1)
    echo "   大小: $SIZE"
    
    # 如果目标不存在，移动并创建符号链接
    if [ ! -d "$CACHE_DIR/modelscope" ] || [ -z "$(ls -A $CACHE_DIR/modelscope 2>/dev/null)" ]; then
        echo "   正在移动..."
        mv /root/.cache/modelscope "$CACHE_DIR/"
        ln -s "$CACHE_DIR/modelscope" /root/.cache/modelscope
        echo "   ✓ ModelScope 缓存已移动到数据盘"
    else
        # 如果目标已存在，只创建符号链接（数据已在数据盘）
        rm -rf /root/.cache/modelscope
        ln -s "$CACHE_DIR/modelscope" /root/.cache/modelscope
        echo "   ✓ 已创建符号链接（数据已在数据盘）"
    fi
    echo ""
fi

# 2. 移动 HuggingFace 缓存（3.6G）
if [ -d "/root/.cache/huggingface" ] && [ ! -L "/root/.cache/huggingface" ]; then
    echo "2. 移动 HuggingFace 缓存 (约 3.6G)..."
    SIZE=$(du -sh /root/.cache/huggingface 2>/dev/null | cut -f1)
    echo "   大小: $SIZE"
    
    if [ ! -d "$CACHE_DIR/huggingface" ] || [ -z "$(ls -A $CACHE_DIR/huggingface 2>/dev/null)" ]; then
        echo "   正在移动..."
        mv /root/.cache/huggingface "$CACHE_DIR/"
        ln -s "$CACHE_DIR/huggingface" /root/.cache/huggingface
        echo "   ✓ HuggingFace 缓存已移动到数据盘"
    else
        rm -rf /root/.cache/huggingface
        ln -s "$CACHE_DIR/huggingface" /root/.cache/huggingface
        echo "   ✓ 已创建符号链接（数据已在数据盘）"
    fi
    echo ""
fi

# 3. 清理其他缓存
echo "3. 清理其他缓存..."
pip cache purge 2>/dev/null && echo "   ✓ 已清理 pip 缓存"
conda clean --all -y 2>/dev/null && echo "   ✓ 已清理 conda 缓存" || echo "   - conda 未安装"
find /root -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && echo "   ✓ 已清理 Python 缓存"
echo ""

# 4. 清理 pip 缓存目录（如果还存在）
if [ -d "/root/.cache/pip" ]; then
    echo "4. 删除 pip 缓存目录..."
    rm -rf /root/.cache/pip
    echo "   ✓ 已删除"
    echo ""
fi

echo "======================================"
echo "清理完成！"
echo "======================================"
echo ""
echo "清理后的系统盘使用情况："
df -h / | tail -1
echo ""
echo "数据盘使用情况："
df -h "$DATA_DISK" | tail -1
echo ""
echo "提示："
echo "  - 模型缓存已移动到数据盘，不会影响程序运行"
echo "  - 符号链接已创建，路径保持一致"
echo "  - 如需更多空间，可以考虑清理 /root/LearningFriend/models 中的旧模型"

