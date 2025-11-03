#!/bin/bash
# 下载 IndexTTS2 所需的 HuggingFace 模型
# 用于离线环境使用

echo "======================================"
echo "下载 IndexTTS2 所需的 HuggingFace 模型"
echo "======================================"

# 检查是否已安装 huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "✗ huggingface-cli 未安装"
    echo "请先安装: pip install huggingface_hub[cli]"
    exit 1
fi

echo ""
echo "开始下载模型..."
echo ""

# 下载所需的模型
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

echo "======================================"
echo "下载完成！"
echo "======================================"
echo ""
echo "模型已保存到 ~/.cache/huggingface/hub/"
echo "现在可以在离线环境中使用 IndexTTS2 了"

