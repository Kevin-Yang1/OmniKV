# 快速运行指南

## RAG 数据集创建

### 在当前主机（推荐先在有足够显存的主机）

```bash
cd /NV1/ykw/projects/OmniKV

# 方式1: 标准运行（需要 GPU，~8-12GB 显存）
CUDA_VISIBLE_DEVICES=0 python scripts/rag/create_narrativeqa_rag.py \
  --input datasets/longbench/narrativeqa.json \
  --output datasets/longbench/narrativeqa_rag \
  --k 10

# 方式2: 低显存模式
./scripts/rag/run_rag_lowmem.sh

# 方式3: CPU 模式（慢但稳定）
python scripts/rag/create_narrativeqa_rag.py \
  --input datasets/longbench/narrativeqa.json \
  --output datasets/longbench/narrativeqa_rag \
  --k 10 \
  --device cpu

# 验证结果
python scripts/rag/validate_narrativeqa_rag.py \
  --rag datasets/longbench/narrativeqa_rag/narrativeqa_rag_k10.json \
  --original datasets/longbench/narrativeqa.json
```

## 迁移到新主机

### Step 1: 准备传输

```bash
cd /NV1/ykw/projects/OmniKV

# 配置 SSH 免密登录
ssh-copy-id ykw@222.20.98.71

# 运行完整迁移
./scripts/migration/migrate_to_new_host.sh
```

### Step 2: 在新主机上设置环境

```bash
# 登录新主机
ssh ykw@222.20.98.71

cd /data/ykw/project/OmniKV

# 设置 conda 环境
./scripts/migration/setup_env_new_host.sh

# 验证环境
conda activate cachecraft
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Step 3: 在新主机运行 RAG 转换

```bash
# 在新主机上
cd /data/ykw/project/OmniKV

# 运行转换
CUDA_VISIBLE_DEVICES=0 python scripts/rag/create_narrativeqa_rag.py \
  --input datasets/longbench/narrativeqa.json \
  --output datasets/longbench/narrativeqa_rag \
  --k 10
```

## 测试和调试

```bash
# 测试 chunking 策略
python scripts/tests/test_chunking.py

# 检查 GPU 状态
nvidia-smi

# 测试环境
python -c "from transformers import AutoModel; print('OK')"
```

## 文件位置

- **RAG 脚本**: `scripts/rag/`
- **迁移脚本**: `scripts/migration/`
- **测试脚本**: `scripts/tests/`
- **文档**: `docs/`

详见 [scripts/README.md](scripts/README.md)
