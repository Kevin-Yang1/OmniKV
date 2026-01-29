# NarrativeQA RAG 数据集创建指南

## 快速开始

```bash
cd /NV1/ykw/projects/OmniKV

# 创建 RAG 数据集 (k=10)
CUDA_VISIBLE_DEVICES=0 python create_narrativeqa_rag.py \
  --input datasets/longbench/narrativeqa.json \
  --output datasets/longbench/narrativeqa_rag \
  --k 10 \
  --chunk_size 1000 \
  --overlap 200

# 验证结果
python validate_narrativeqa_rag.py \
  --rag datasets/longbench/narrativeqa_rag/narrativeqa_rag_k10.json \
  --original datasets/longbench/narrativeqa.json
```

## 参数说明

### create_narrativeqa_rag.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必需 | 原始 narrativeqa.json 路径 |
| `--output` | 必需 | 输出目录 |
| `--k` | 10 | 检索的 chunk 数量 |
| `--chunk_size` | 1000 | Chunk 大小（字符） |
| `--overlap` | 200 | Chunk 重叠大小 |
| `--model` | nvidia/NV-Embed-v2 | 嵌入模型 |
| `--device` | cuda | 计算设备 |
| `--batch_size` | 32 | 嵌入批次大小 |

## 输出文件

```
datasets/longbench/narrativeqa_rag/
├── narrativeqa_rag_k10.json      # RAG 数据集
├── chunk_embeddings.pt            # Chunk 向量 (PyTorch)
├── question_embeddings.pt         # 问题向量 (PyTorch)
├── chunk_metadata.json            # Chunk 元数据
└── processing_log.json            # 处理日志
```

## 数据格式

### 输入 (原始 narrativeqa.json)

```json
{
    "input": "What is Saltram's living situation?",
    "context": ["long novel text..."],
    "answers": ["He is a guest in the home of the Mulvilles."],
    "length": 22692,
    ...
}
```

### 输出 (narrativeqa_rag_k10.json)

```json
{
    "input": "What is Saltram's living situation?",
    "context": [
        "chunk1 (most relevant)...",
        "chunk2...",
        ...
        "chunk10 (least relevant)..."
    ],
    "answers": ["He is a guest in the home of the Mulvilles."],
    "length": 9824,
    "metadata": {
        "original_length": 22692,
        "retrieval_k": 10,
        "chunk_scores": [0.95, 0.89, ...],
        "chunk_indices": [42, 15, ...]
    }
}
```

## 高级用法

### 不同的 k 值

```bash
# 生成多个版本
for k in 5 10 15 20; do
    python create_narrativeqa_rag.py \
      --input datasets/longbench/narrativeqa.json \
      --output datasets/longbench/narrativeqa_rag \
      --k $k
done
```

### 使用不同的嵌入模型

```bash
# 使用其他模型 (需要先下载)
python create_narrativeqa_rag.py \
  --input datasets/longbench/narrativeqa.json \
  --output datasets/longbench/narrativeqa_rag_bge \
  --model BAAI/bge-large-en-v1.5 \
  --k 10
```

### CPU 模式（显存不足时）

```bash
python create_narrativeqa_rag.py \
  --input datasets/longbench/narrativeqa.json \
  --output datasets/longbench/narrativeqa_rag \
  --device cpu \
  --batch_size 16  # 降低批次大小
```

## 预期时间

- **Context 提取**: ~1 分钟
- **Chunking**: ~2 分钟
- **Chunk 嵌入**: ~10-15 分钟 (取决于 GPU)
- **问题嵌入**: ~1 分钟
- **检索**: ~2 分钟
- **总计**: ~20-30 分钟

## 资源需求

- **显存**: 8-12 GB (NV-Embed-v2 模型)
- **内存**: 4-8 GB
- **磁盘**: ~500 MB (向量文件 + 数据集)

## 故障排查

### OOM (显存不足)

```bash
# 降低批次大小
python create_narrativeqa_rag.py ... --batch_size 16

# 或使用 CPU
python create_narrativeqa_rag.py ... --device cpu
```

### 模型下载失败

```bash
# 先手动下载模型
huggingface-cli download nvidia/NV-Embed-v2

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

## 质量验证

运行验证脚本检查：

```bash
python validate_narrativeqa_rag.py \
  --rag datasets/longbench/narrativeqa_rag/narrativeqa_rag_k10.json \
  --original datasets/longbench/narrativeqa.json
```

**验证内容**:
- ✓ 样本数量一致性
- ✓ ID/问题/答案匹配
- ✓ Chunk 数量统计
- ✓ 长度压缩比
- ✓ 检索分数分布

## 在 LongBench 上测试

```bash
# 使用 RAG 数据集运行 CacheCraft
PYTHONPATH=. python baselines/cachecraft/run_longbench.py \
  --data_path datasets/longbench/narrativeqa_rag/narrativeqa_rag_k10.json \
  --model_path /path/to/model \
  --run_all_modes

# 评测
python benchmark/long_bench/eval.py --model cachecraft --cfg recompute
```

## 注意事项

1. **首次运行**: 需要下载 NV-Embed-v2 模型 (~1-2 GB)
2. **相似度**: 检索分数应 > 0.5，否则可能需要调整参数
3. **Chunk 大小**: 太小会损失上下文，太大会引入噪音
4. **重叠**: 建议 15-25% 的重叠以保持连贯性
