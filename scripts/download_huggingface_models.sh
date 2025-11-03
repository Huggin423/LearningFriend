#!/bin/bash
# 下载 IndexTTS2 所需的 HuggingFace 模型
# 用于离线环境使用
# 支持国内镜像站点

echo "======================================"
echo "下载 IndexTTS2 所需的 HuggingFace 模型"
echo "======================================"

# 检查是否已安装 huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "✗ huggingface-cli 未安装"
    echo "请先安装: pip install huggingface_hub[cli]"
    exit 1
fi

# 提示用户选择镜像
echo ""
echo "请选择下载方式："
echo "  1) HuggingFace (国际用户)"
echo "  2) ModelScope (国内镜像，但campplus不可用)"
echo "  3) HF-Mirror (国内HuggingFace镜像，推荐国内用户)"
read -p "请输入选项 (1/2/3，默认3): " choice
choice=${choice:-3}

# 设置镜像环境变量
if [ "$choice" == "2" ]; then
    echo ""
    echo "使用 ModelScope 镜像下载..."
    echo "提示：ModelScope 为国内镜像，下载速度更快"
    
    # 使用ModelScope下载（只下载ModelScope上有的模型）
    models_ms=(
        "facebook/w2v-bert-2.0"
        "amphion/MaskGCT"
    )
    
    # 先从ModelScope下载
    for model in "${models_ms[@]}"; do
        echo ""
        echo "正在从ModelScope下载: $model"
        python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
import os
model='$model'
print(f'下载 {model} 到缓存目录...')
try:
    path = snapshot_download(model)
    print(f'✓ 下载成功: {path}')
except Exception as e:
    print(f'✗ 下载失败: {e}')
"
    done
    
    # 提示：campplus需要使用HuggingFace
    echo ""
    echo "注意: funasr/campplus 在ModelScope上不可用"
    echo "建议使用选项1或3从HuggingFace下载此模型"
    
elif [ "$choice" == "3" ]; then
    echo ""
    echo "使用 HF-Mirror 镜像下载..."
    export HF_ENDPOINT=https://hf-mirror.com
    
    models=(
        "facebook/w2v-bert-2.0"
        "amphion/MaskGCT"
        "funasr/campplus"
    )
    
    for model in "${models[@]}"; do
        echo "正在下载: $model"
        huggingface-cli download "$model" --local-dir-use-symlinks False
        if [ $? -eq 0 ]; then
            echo "✓ $model 下载完成"
        else
            echo "✗ $model 下载失败"
        fi
        echo ""
    done
    
else
    echo ""
    echo "使用 HuggingFace 官方站点下载..."
    
    models=(
        "facebook/w2v-bert-2.0"
        "amphion/MaskGCT"
        "funasr/campplus"
    )
    
    for model in "${models[@]}"; do
        echo "正在下载: $model"
        huggingface-cli download "$model" --local-dir-use-symlinks False
        if [ $? -eq 0 ]; then
            echo "✓ $model 下载完成"
        else
            echo "✗ $model 下载失败"
        fi
        echo ""
    done
fi

echo "======================================"
echo "下载完成！"
echo "======================================"
echo ""
echo "模型已保存到缓存目录"
echo "现在可以在离线环境中使用 IndexTTS2 了"

