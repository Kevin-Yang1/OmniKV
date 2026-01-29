#!/bin/bash
# 仅传输模型文件脚本（可选）
# 如果新机器无法直接从 HuggingFace 下载模型，使用此脚本传输

set -e

NEW_HOST="222.20.98.71"
MODEL_SRC="/NV1/ykw/models/Meta-Llama-3.1-8B-Instruct"
MODEL_DEST="/data/ykw/models"

echo "========================================"
echo "传输模型文件"
echo "========================================"
echo "源: $MODEL_SRC"
echo "目标: ykw@$NEW_HOST:$MODEL_DEST/"
echo ""
echo "提示: 模型文件较大（约16GB），可能需要较长时间..."
echo ""

# 在新机器创建目录
ssh ykw@$NEW_HOST "mkdir -p $MODEL_DEST"

# 使用 rsync 传输（支持断点续传）
rsync -avz --progress --info=progress2 \
  $MODEL_SRC/ \
  ykw@$NEW_HOST:$MODEL_DEST/Meta-Llama-3.1-8B-Instruct/

echo ""
echo "✅ 模型传输完成！"
echo "模型路径: $MODEL_DEST/Meta-Llama-3.1-8B-Instruct"
