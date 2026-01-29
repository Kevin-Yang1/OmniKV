#!/bin/bash
# OmniKV 项目迁移脚本
# 用途：将当前环境完整迁移到新服务器

set -e  # 遇到错误立即退出

# ============ 配置区 ============
NEW_HOST="222.20.98.71"
DEST_PATH="/data/ykw/project/OmniKV"
CURRENT_PATH="/NV1/ykw/projects/OmniKV"

echo "========================================"
echo "OmniKV 环境迁移脚本"
echo "========================================"
echo "源路径: $CURRENT_PATH"
echo "目标主机: $NEW_HOST"
echo "目标路径: $DEST_PATH"
echo ""

# ============ Step 1: 导出环境 ============
echo "[1/5] 导出 Conda 环境..."
cd $CURRENT_PATH
conda env export > environment.yml
pip list --format=freeze > requirements.txt
echo "✓ 环境文件已生成: environment.yml, requirements.txt"

# ============ Step 2: 在新机器创建目录 ============
echo ""
echo "[2/5] 在新机器创建目录..."
ssh ykw@$NEW_HOST "mkdir -p $DEST_PATH"
echo "✓ 目录已创建: $DEST_PATH"

# ============ Step 3: 传输代码和配置文件 ============
echo ""
echo "[3/5] 传输项目代码和配置..."
rsync -avz --progress \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='.pytest_cache' \
  --exclude='.DS_Store' \
  --exclude='datasets/*' \
  --exclude='output/*' \
  --exclude='models/*' \
  baselines/ \
  benchmark/ \
  modeling/ \
  *.py \
  *.md \
  *.txt \
  environment.yml \
  requirements.txt \
  ykw@$NEW_HOST:$DEST_PATH/

echo "✓ 代码文件传输完成"

# ============ Step 4: 传输数据集（解引用软链接） ============
echo ""
echo "[4/5] 传输数据集（跟随软链接）..."
echo "提示: 这可能需要较长时间，取决于数据集大小..."

# -L: 跟随软链接，传输实际文件
# --info=progress2: 显示总体进度
rsync -avzL --progress --info=progress2 \
  datasets/ \
  ykw@$NEW_HOST:$DEST_PATH/datasets/

echo "✓ 数据集传输完成"

# ============ Step 5: 传输输出目录 ============
echo ""
echo "[5/5] 传输输出目录..."

# 检查 output 目录是否存在
if [ -d "$CURRENT_PATH/baselines/cachecraft/output" ]; then
    # 先在新机器创建目标目录
    ssh ykw@$NEW_HOST "mkdir -p $DEST_PATH/baselines/cachecraft"
    
    rsync -avz --progress \
      baselines/cachecraft/output/ \
      ykw@$NEW_HOST:$DEST_PATH/baselines/cachecraft/output/
    echo "✓ 输出目录传输完成"
else
    echo "⚠️  输出目录不存在，跳过"
fi

# ============ 在新机器上设置环境 ============
echo ""
echo "========================================"
echo "在新机器上设置 Conda 环境..."
echo "========================================"

ssh ykw@$NEW_HOST << 'ENDSSH'
set -e

# 配置
DEST_PATH="/data/ykw/project/OmniKV"
ENV_NAME="cachecraft"

cd $DEST_PATH

echo "1. 检查 Conda..."
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Miniconda 或 Anaconda"
    exit 1
fi

echo "2. 创建 Conda 环境..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "环境 $ENV_NAME 已存在，删除并重新创建..."
    conda env remove -n $ENV_NAME -y
fi

conda env create -f environment.yml -n $ENV_NAME

echo "3. 验证环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

python << 'ENDPYTHON'
import sys
print(f"Python 版本: {sys.version}")

try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("警告: PyTorch 未正确安装")
    
try:
    import transformers
    print(f"Transformers 版本: {transformers.__version__}")
except ImportError:
    print("警告: Transformers 未正确安装")
ENDPYTHON

echo ""
echo "✓ 环境设置完成！"
echo ""
echo "使用以下命令激活环境："
echo "  cd $DEST_PATH"
echo "  conda activate $ENV_NAME"

ENDSSH

# ============ 完成 ============
echo ""
echo "========================================"
echo "✅ 迁移完成！"
echo "========================================"
echo ""
echo "下一步操作："
echo "1. 登录到新机器:"
echo "   ssh ykw@$NEW_HOST"
echo ""
echo "2. 进入项目目录:"
echo "   cd $DEST_PATH"
echo ""
echo "3. 激活环境:"
echo "   conda activate cachecraft"
echo ""
echo "4. 传输模型文件（如果需要）:"
echo "   在新机器上下载或从当前机器传输 /NV1/ykw/models/Meta-Llama-3.1-8B-Instruct"
echo ""
echo "5. 测试运行:"
echo "   PYTHONPATH=. python baselines/cachecraft/run_longbench.py \\"
echo "     --data_path datasets/longbench/2wikimqa.json \\"
echo "     --model_path /path/to/model \\"
echo "     --num_samples 3"
echo ""
