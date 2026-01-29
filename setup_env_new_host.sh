#!/bin/bash
# 新机器环境安装脚本（分步）
# 解决 flash-attn 需要 torch 的依赖问题

set -e

ENV_NAME="cachecraft"

echo "========================================"
echo "CacheCraft 环境安装（新机器）"
echo "========================================"

# Step 1: 创建基础环境
echo "[1/4] 创建基础 conda 环境..."
conda create -n $ENV_NAME python=3.10 -y

# Step 2: 激活环境
echo "[2/4] 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Step 3: 先安装 PyTorch 和基础依赖
echo "[3/4] 安装 PyTorch 和基础依赖..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install numpy packaging ninja

# Step 4: 安装其他依赖（从 requirements.txt，跳过 flash-attn）
echo "[4/4] 安装其他依赖..."
pip install einops
pip install jieba rouge nltk
pip install aiohttp aiohappyeyeballs aiosignal
pip install antlr4-python3-runtime
pip install anyio async-timeout attrs
pip install certifi charset-normalizer
pip install dill exceptiongroup filelock

# Step 5: （可选）编译安装 flash-attn
echo ""
echo "========================================"
echo "Flash Attention 安装（可选）"
echo "========================================"
read -p "是否安装 flash-attn？这需要编译，可能需要 10-20 分钟 (y/n): " install_flash

if [ "$install_flash" = "y" ] || [ "$install_flash" = "Y" ]; then
    echo "安装 flash-attn..."
    pip install flash-attn --no-build-isolation
    echo "✓ flash-attn 安装完成"
else
    echo "⚠️  跳过 flash-attn 安装（CacheCraft 可以不使用它运行）"
fi

echo ""
echo "========================================"
echo "✅ 环境安装完成！"
echo "========================================"

# 验证安装
echo ""
echo "验证安装..."
python << 'ENDPYTHON'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

import transformers
print(f"Transformers: {transformers.__version__}")

try:
    import flash_attn
    print(f"Flash Attention: {flash_attn.__version__}")
except ImportError:
    print("Flash Attention: 未安装（可选）")
ENDPYTHON

echo ""
echo "使用以下命令激活环境："
echo "  conda activate $ENV_NAME"
