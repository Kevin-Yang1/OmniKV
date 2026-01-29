# OmniKV 项目路径配置

## 问题

在不同主机上，基础路径不同：
- **原主机**: `/NV1/ykw/`
- **新主机**: `/data/ykw/`

每次都需要手动修改路径很麻烦。

## 解决方案

使用配置文件自动检测和适配路径。

---

## 方案 A: Shell 脚本配置（推荐用于 bash 脚本）

### 1. 使用配置文件

所有 shell 脚本都应该先加载配置：

```bash
#!/bin/bash
# 加载配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

# 使用配置的路径
python scripts/rag/create_narrativeqa_rag.py \
  --input "${OMNIKV_DATASETS}/longbench/narrativeqa.json" \
  --output "${OMNIKV_DATASETS}/longbench/narrativeqa_rag"
```

### 2. 配置文件说明

[scripts/config.sh](file:///NV1/ykw/projects/OmniKV/scripts/config.sh) 会自动检测：

```bash
# 自动检测
if [ -d "/NV1/ykw" ]; then
    export OMNIKV_BASE="/NV1/ykw"
elif [ -d "/data/ykw" ]; then
    export OMNIKV_BASE="/data/ykw"
fi

# 设置派生路径
export OMNIKV_PROJECT="${OMNIKV_BASE}/projects/OmniKV"
export OMNIKV_MODELS="${OMNIKV_BASE}/models"
```

### 3. 可用变量

- `OMNIKV_BASE`: 基础路径 (`/NV1/ykw` 或 `/data/ykw`)
- `OMNIKV_PROJECT`: 项目路径
- `OMNIKV_MODELS`: 模型目录
- `OMNIKV_DATASETS`: 数据集目录
- `OMNIKV_OUTPUT`: 输出目录

---

## 方案 B: Python 配置（推荐用于 Python 脚本）

### 1. 导入配置模块

```python
import sys
sys.path.insert(0, '/path/to/OmniKV')  # 或使用相对路径

from scripts.omnikv_config import config

# 使用配置
parser.add_argument(
    '--input',
    default=str(config.get_dataset_path('narrativeqa.json'))
)

parser.add_argument(
    '--model',
    default=config.get_model_path('NV-Embed-v2')
)
```

### 2. 配置模块功能

```python
from scripts.omnikv_config import config

# 检查主机类型
print(config.host_type)  # 'original' 或 'new'

# 获取路径
config.project_root      # Path('/NV1/ykw/projects/OmniKV')
config.models_dir        # Path('/NV1/ykw/models')
config.datasets_dir      # Path('/NV1/ykw/projects/OmniKV/datasets')

# 工具方法
config.get_dataset_path('narrativeqa.json')
config.get_model_path('NV-Embed-v2')
```

---

## 方案 C: 环境变量（手动设置）

如果自动检测不工作，可以手动设置：

### Bash (~/.bashrc)

```bash
# 在原主机
export OMNIKV_BASE="/NV1/ykw"

# 在新主机
export OMNIKV_BASE="/data/ykw"
```

### Python 脚本启动时

```bash
OMNIKV_BASE=/data/ykw python scripts/rag/create_narrativeqa_rag.py ...
```

---

## 使用示例

### 更新后的脚本

已更新的脚本（自动使用配置）：
- ✅ `scripts/rag/run_rag_lowmem.sh` - 已更新

需要更新的脚本（可选）：
- `scripts/migration/*.sh` - 迁移脚本（路径是参数，不需要）
- 其他自定义脚本

### 在新主机上

```bash
# 1. 传输项目到新主机
# ... (使用迁移脚本)

# 2. 登录新主机
ssh ykw@222.20.98.71
cd /data/ykw/project/OmniKV

# 3. 测试配置
source scripts/config.sh
echo $OMNIKV_BASE  # 应该显示 /data/ykw

# 或测试 Python 配置
python -c "from scripts.omnikv_config import config; print(config)"

# 4. 直接运行脚本（自动适配）
./scripts/rag/run_rag_lowmem.sh  # 会自动使用 /data/ykw
```

---

## 最佳实践

### 新建脚本时

**Shell 脚本**:
```bash
#!/bin/bash
source "$(dirname "$0")/../config.sh"  # 加载配置
# ... 使用 $OMNIKV_* 变量
```

**Python 脚本**:
```python
from scripts.omnikv_config import config
# ... 使用 config.* 属性
```

### 命令行参数

仍然允许通过参数覆盖：
```bash
python script.py --input /custom/path/data.json  # 覆盖默认
```

---

## 测试配置

```bash
# Shell 配置
source scripts/config.sh

# Python 配置
python scripts/omnikv_config.py
```

预期输出：
```
OmniKVConfig(
  host_type=new
  base_path=/data/ykw
  project_root=/data/ykw/project/OmniKV
  ...
)
```
