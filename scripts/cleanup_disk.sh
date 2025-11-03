#!/bin/bash
# 清理磁盘空间脚本
# 用于清理 autodl 系统中的临时文件和缓存

echo "======================================"
echo "磁盘清理脚本"
echo "======================================"
echo ""

# 显示当前磁盘使用情况
echo "当前磁盘使用情况："
df -h / | tail -1
echo ""

# 计算可以清理的空间
echo "正在检查可清理的内容..."
echo ""

total_freed=0

# 1. 清理 pip 缓存
echo "1. 清理 pip 缓存..."
pip_cache_size=$(du -sh /root/.cache/pip 2>/dev/null | cut -f1)
if [ -d "/root/.cache/pip" ]; then
    pip cache purge 2>/dev/null
    echo "   ✓ 已清理 pip 缓存 (约 $pip_cache_size)"
    total_freed=$((total_freed + 1))
else
    echo "   - pip 缓存目录不存在"
fi
echo ""

# 2. 清理 conda 缓存
echo "2. 清理 conda 缓存..."
if command -v conda &> /dev/null; then
    conda clean --all -y 2>/dev/null
    echo "   ✓ 已清理 conda 缓存"
    total_freed=$((total_freed + 1))
else
    echo "   - conda 未安装"
fi
echo ""

# 3. 清理 Python __pycache__ 和 .pyc 文件
echo "3. 清理 Python 缓存文件..."
pycache_size=$(find /root -type d -name "__pycache__" -exec du -sh {} \; 2>/dev/null | awk '{s+=$1} END {print s}')
if [ -n "$pycache_size" ] && [ "$pycache_size" != "0" ]; then
    find /root -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find /root -name "*.pyc" -delete 2>/dev/null
    find /root -name "*.pyo" -delete 2>/dev/null
    echo "   ✓ 已清理 Python 缓存文件"
    total_freed=$((total_freed + 1))
else
    echo "   - 未找到 Python 缓存文件"
fi
echo ""

# 4. 清理临时文件
echo "4. 清理临时文件..."
if [ -d "/tmp" ]; then
    tmp_size=$(du -sh /tmp 2>/dev/null | cut -f1)
    # 只清理 7 天前的文件
    find /tmp -type f -atime +7 -delete 2>/dev/null
    echo "   ✓ 已清理 /tmp 中 7 天前的文件 (约 $tmp_size)"
    total_freed=$((total_freed + 1))
fi
echo ""

# 5. 清理日志文件
echo "5. 清理日志文件..."
if [ -d "/var/log" ]; then
    log_size=$(find /var/log -name "*.log" -type f -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)
    find /var/log -name "*.log" -type f -size +100M -delete 2>/dev/null
    echo "   ✓ 已清理大型日志文件 (约 $log_size)"
    total_freed=$((total_freed + 1))
fi
echo ""

# 6. 检查是否有重复的模型文件（在 .cache 目录中）
echo "6. 检查模型缓存..."
echo "   提示：模型文件较大，请手动决定是否清理"
echo "   - /root/.cache/modelscope: $(du -sh /root/.cache/modelscope 2>/dev/null | cut -f1)"
echo "   - /root/.cache/huggingface: $(du -sh /root/.cache/huggingface 2>/dev/null | cut -f1)"
echo "   如需清理，请运行："
echo "     rm -rf /root/.cache/pip"
echo "     # 注意：清理模型缓存可能会影响程序运行"
echo ""

# 显示清理后的磁盘使用情况
echo "======================================"
echo "清理完成！"
echo "======================================"
echo ""
echo "清理后的磁盘使用情况："
df -h / | tail -1
echo ""
echo "已清理 $total_freed 项内容"
echo ""
echo "提示：如果需要更多空间，可以考虑："
echo "  1. 将模型文件移动到 /autodl-pub (数据盘，有更多空间)"
echo "  2. 清理不再使用的模型缓存"
echo "  3. 删除旧的实验数据"

