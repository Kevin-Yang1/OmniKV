#!/bin/bash
# 低显存模式运行 RAG 转换
# 使用配置文件自动适配路径

set -e

# 加载配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "========================================"
echo "RAG 转换 - 低显存模式"
echo "========================================"

# 清理 GPU 进程
echo "清理 GPU 进程..."
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true

sleep 2

echo "检查 GPU 状态..."
nvidia-smi

echo ""
echo "运行 RAG 转换..."
echo "  输入: ${OMNIKV_DATASETS}/longbench/narrativeqa.json"
echo "  输出: ${OMNIKV_DATASETS}/longbench/narrativeqa_rag"
echo ""

cd "${OMNIKV_PROJECT}"

CUDA_VISIBLE_DEVICES=0 python scripts/rag/create_narrativeqa_rag.py \
  --input "${OMNIKV_DATASETS}/longbench/narrativeqa.json" \
  --output "${OMNIKV_DATASETS}/longbench/narrativeqa_rag" \
  --k 10 \
  --batch_size 8 \
  --chunk_size 1000 \
  --overlap 200
