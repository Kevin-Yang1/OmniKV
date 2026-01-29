#!/bin/bash
# OmniKV 项目配置文件
# 根据主机自动检测或手动设置基础路径

# 方式1: 自动检测（推荐）
if [ -d "/NV1/ykw" ]; then
    export OMNIKV_BASE="/NV1/ykw"
    export OMNIKV_HOST="original"
elif [ -d "/data/ykw" ]; then
    export OMNIKV_BASE="/data/ykw"
    export OMNIKV_HOST="new"
else
    # 默认值（手动设置）
    export OMNIKV_BASE="/NV1/ykw"
    export OMNIKV_HOST="unknown"
fi

# 项目路径
export OMNIKV_PROJECT="${OMNIKV_BASE}/projects/OmniKV"
export OMNIKV_MODELS="${OMNIKV_BASE}/models"

# 数据集路径
export OMNIKV_DATASETS="${OMNIKV_PROJECT}/datasets"

# 输出路径
export OMNIKV_OUTPUT="${OMNIKV_PROJECT}/baselines/cachecraft/output"

# 显示配置
echo "OmniKV Configuration:"
echo "  Host: ${OMNIKV_HOST}"
echo "  Base: ${OMNIKV_BASE}"
echo "  Project: ${OMNIKV_PROJECT}"
echo "  Models: ${OMNIKV_MODELS}"
echo "  Datasets: ${OMNIKV_DATASETS}"
