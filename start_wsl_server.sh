#!/bin/bash
# HeyGem TCP 流式服务端启动脚本 (WSL2侧)
# 自动处理: conda环境 + CUDA 11.8/cuDNN 8.9 库路径

# 激活conda环境 (Python 3.9 + PyTorch 2.0.1 CUDA 11.8)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate py39

# CUDA 11.8 + cuDNN 8.9 库路径 (Pascal CC 6.1 兼容)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$(dirname "$0")"

echo "=== HeyGem TCP Server ==="
echo "Conda: $CONDA_DEFAULT_ENV | Python: $(python --version)"
echo "CUDA libs: $CONDA_PREFIX/lib"
echo "=========================="

python3 stream_server_tcp.py
