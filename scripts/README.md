# OmniKV Scripts

项目辅助脚本集合，按功能分类存放。

## 目录结构

```
scripts/
├── rag/               # RAG 数据集相关脚本
│   ├── create_narrativeqa_rag.py
│   ├── validate_narrativeqa_rag.py
│   └── run_rag_lowmem.sh
├── migration/         # 环境迁移脚本
│   ├── migrate_to_new_host.sh
│   ├── migrate_model.sh
│   └── setup_env_new_host.sh
└── tests/             # 测试脚本
    └── test_chunking.py
```

## RAG 数据集脚本 (`rag/`)

### create_narrativeqa_rag.py
创建 NarrativeQA RAG 版本数据集

**用法**:
```bash
cd /path/to/OmniKV
python scripts/rag/create_narrativeqa_rag.py \
  --input datasets/longbench/narrativeqa.json \
  --output datasets/longbench/narrativeqa_rag \
  --k 10
```

**说明**: 详见 [docs/narrativeqa_rag_guide.md](../docs/narrativeqa_rag_guide.md)

### validate_narrativeqa_rag.py
验证 RAG 数据集质量

**用法**:
```bash
python scripts/rag/validate_narrativeqa_rag.py \
  --rag datasets/longbench/narrativeqa_rag/narrativeqa_rag_k10.json \
  --original datasets/longbench/narrativeqa.json
```

### run_rag_lowmem.sh
低显存模式运行 RAG 转换（清理 GPU + 小 batch）

**用法**:
```bash
./scripts/rag/run_rag_lowmem.sh
```

## 迁移脚本 (`migration/`)

### migrate_to_new_host.sh
完整项目迁移（代码、数据、环境）

**用法**:
```bash
# 修改脚本中的目标主机地址
./scripts/migration/migrate_to_new_host.sh
```

### migrate_model.sh
迁移大型模型文件

**用法**:
```bash
./scripts/migration/migrate_model.sh
```

### setup_env_new_host.sh
在新主机上设置 Conda 环境

**用法**:
```bash
# 在新主机上运行
./scripts/migration/setup_env_new_host.sh
```

## 测试脚本 (`tests/`)

### test_chunking.py
测试递归字符分割策略

**用法**:
```bash
python scripts/tests/test_chunking.py
```

## 快速索引

| 任务 | 脚本 |
|------|------|
| 创建 RAG 数据集 | `scripts/rag/create_narrativeqa_rag.py` |
| 验证 RAG 数据集 | `scripts/rag/validate_narrativeqa_rag.py` |
| 迁移到新主机 | `scripts/migration/migrate_to_new_host.sh` |
| 设置新环境 | `scripts/migration/setup_env_new_host.sh` |
| 测试切片策略 | `scripts/tests/test_chunking.py` |
