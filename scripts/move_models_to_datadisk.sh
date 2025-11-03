#!/bin/bash
# 将模型文件移动到数据盘（/autodl-pub）以节省系统盘空间
# 数据盘有更多空间（1.6T 可用）

echo "======================================"
echo "移动模型到数据盘"
echo "======================================"
echo ""

# 检查数据盘空间
echo "数据盘空间："
df -h /autodl-pub | tail -1
echo ""

DATADISK_DIR="/autodl-pub/models_cache"
mkdir -p "$DATADISK_DIR"

# 方案1: 创建符号链接（推荐，节省空间且保持路径一致）
echo "方案1: 创建符号链接（推荐）"
echo "  将模型缓存移动到数据盘，并创建符号链接"
echo ""

read -p "是否移动模型缓存到数据盘？ (y/N): " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "已取消"
    exit 0
fi

# 移动 huggingface 缓存
if [ -d "/root/.cache/huggingface" ]; then
    echo "移动 HuggingFace 缓存..."
    hf_size=$(du -sh /root/.cache/huggingface | cut -f1)
    echo "  当前大小: $hf_size"
    
    # 备份到数据盘
    if [ ! -d "$DATADISK_DIR/huggingface" ]; then
        mv /root/.cache/huggingface "$DATADISK_DIR/"
        ln -s "$DATADISK_DIR/huggingface" /root/.cache/huggingface
        echo "  ✓ 已移动 HuggingFace 缓存并创建符号链接"
    else
        echo "  ⚠ 目标目录已存在，跳过"
    fi
fi

# 移动 modelscope 缓存（更大，但需要更多考虑）
read -p "是否移动 ModelScope 缓存（17G）到数据盘？ (y/N): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    if [ -d "/root/.cache/modelscope" ]; then
        echo "移动 ModelScope 缓存..."
        ms_size=$(du -sh /root/.cache/modelscope | cut -f1)
        echo "  当前大小: $ms_size"
        
        if [ ! -d "$DATADISK_DIR/modelscope" ]; then
            mv /root/.cache/modelscope "$DATADISK_DIR/"
            ln -s "$DATADISK_DIR/modelscope" /root/.cache/modelscope
            echo "  ✓ 已移动 ModelScope 缓存并创建符号链接"
        else
            echo "  ⚠ 目标目录已存在，跳过"
        fi
    fi
fi

echo ""
echo "======================================"
echo "完成！"
echo "======================================"
echo ""
echo "清理后的系统盘使用情况："
df -h / | tail -1
echo ""
echo "数据盘使用情况："
df -h /autodl-pub | tail -1

