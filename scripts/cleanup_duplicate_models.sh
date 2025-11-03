#!/bin/bash
# 清理重复的模型文件

echo "======================================"
echo "清理重复模型文件"
echo "======================================"
echo ""

# 检查 HuggingFace 缓存中的重复 snapshot
echo "检查 HuggingFace 缓存中的重复文件..."

# facebook/w2v-bert-2.0 有两个 snapshot，一个是符号链接，一个是实际下载
w2v_dir="/root/.cache/huggingface/hub/models--facebook--w2v-bert-2.0/snapshots"
if [ -d "$w2v_dir" ]; then
    echo "检查 $w2v_dir:"
    for snapshot in "$w2v_dir"/*; do
        if [ -L "$snapshot" ]; then
            echo "  - $(basename $snapshot): 符号链接 -> $(readlink $snapshot)"
        elif [ -d "$snapshot" ]; then
            size=$(du -sh "$snapshot" 2>/dev/null | cut -f1)
            echo "  - $(basename $snapshot): 目录 ($size)"
        fi
    done
    
    # 如果存在重复的 snapshot（符号链接指向的源文件也在 snapshot 中），可以清理
    read -p "是否清理重复的 snapshot 目录？ (y/N): " answer
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        # 保留符号链接，删除实际下载的重复目录
        for snapshot in "$w2v_dir"/*; do
            if [ -d "$snapshot" ] && [ ! -L "$snapshot" ]; then
                echo "  删除重复的 snapshot: $(basename $snapshot)"
                rm -rf "$snapshot"
            fi
        done
        echo "  ✓ 已清理重复文件"
    fi
fi

echo ""
echo "完成！"

