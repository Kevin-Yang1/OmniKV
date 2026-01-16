#!/bin/bash
# 激活环境 (如果需要)
# source activate cachecraft

# 设置 python 路径
export PYTHONPATH=./

# 运行 HotpotQA 测试
# 请根据实际情况调整 data_path 和 model_path
python baselines/cachecraft/run_hotpotqa.py \
    --data_path datasets/hotpotqa/hotpot_dev_distractor_v1.json \
    --model_path /NV1/ykw/models/Meta-Llama-3.1-8B-Instruct \
    --num_samples 1 \
    --device cuda
