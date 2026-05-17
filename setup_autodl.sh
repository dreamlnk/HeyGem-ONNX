#!/bin/bash
# ============================================================
# AutoDL 一键部署脚本
# 用法: bash setup_autodl.sh
# ============================================================
set -e

echo "============================================================"
echo "  HeyGem Wav2Lip — AutoDL 部署"
echo "============================================================"

# --- 1. CUDA 环境检测 ---
echo ""
echo "[1/5] 检测 GPU 环境..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "PyTorch 未安装，将安装..."

# --- 2. 安装系统包 + Python依赖 ---
echo ""
echo "[2/5] 安装系统包 + Python 依赖..."
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 2>/dev/null || true

# AutoDL 通常已装 PyTorch，检查是否需要重装
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
    echo "安装 PyTorch CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# onnxruntime-gpu 换成 onnxruntime (避免 CUDA 版本冲突)
pip install -q numpy==1.23.5 opencv-python-headless==4.11.0.86 librosa==0.11.0 \
    soundfile==0.13.1 scipy==1.13.1 matplotlib==3.9.4 \
    onnxruntime==1.16.0 tqdm einops scikit-image

echo "依赖安装完成"

# --- 3. 检查模型权重 ---
echo ""
echo "[3/5] 检查模型权重..."
PRETRAIN_DIR="./pretrain_models"
mkdir -p "$PRETRAIN_DIR"

if [ -f "$PRETRAIN_DIR/wav2lip_gan.pth" ]; then
    echo "  ✓ wav2lip_gan.pth ($(du -h $PRETRAIN_DIR/wav2lip_gan.pth | cut -f1))"
else
    echo "  ✗ wav2lip_gan.pth 缺失! 请从本地上传 (435MB)"
fi

if [ -f "$PRETRAIN_DIR/wav2lip256.pth" ]; then
    echo "  ✓ wav2lip256.pth ($(du -h $PRETRAIN_DIR/wav2lip256.pth | cut -f1))"
else
    echo "  ✗ wav2lip256.pth 缺失! 请从本地上传 (214MB)"
fi

# 检查 YuNet 人脸检测模型
if [ -f "$PRETRAIN_DIR/checkpoints/face_detection_yunet_2023mar.onnx" ]; then
    echo "  ✓ YuNet ONNX"
else
    echo "  ✗ YuNet ONNX 缺失，将从 OpenCV 自动下载"
fi

# --- 4. 测试导入 ---
echo ""
echo "[4/5] 测试 Python 导入..."
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
import torch
import cv2
print('  ✓ numpy, torch, cv2')

from phase4_wav2lip import Wav2LipInferenceEngine
print('  ✓ Wav2LipInferenceEngine')

from pipeline_wav2lip import StreamingPipeline
print('  ✓ StreamingPipeline')

from phase1_scrfd_test import scrfd_load, scrfd_detect
print('  ✓ YuNet detection')

from phase2_audio_wav2lip import mel_spectrogram, get_wav2lip_mel_input
print('  ✓ Mel spectrogram')
" || {
    echo ""
    echo "导入失败! 检查是否有文件缺失"
    exit 1
}

# --- 5. 快速预热测试 ---
echo ""
echo "[5/5] 快速预热测试 (加载模型 + 推理 1 帧)..."
python -c "
import sys, time, numpy as np
sys.path.insert(0, '.')
from pipeline_wav2lip import StreamingPipeline

print('  加载 256×256 管线...')
t0 = time.time()
p = StreamingPipeline(detect_interval=4, test_audio=False, size=256, use_align=False)
p.start()
print(f'  加载耗时: {time.time()-t0:.1f}s')

# 模拟一帧
dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
dummy_audio = np.random.randn(1600).astype(np.float32) * 0.01
p.feed_audio(dummy_audio)

t1 = time.time()
rendered, coords = p.process_frame(dummy)
t2 = time.time()
if rendered is not None:
    print(f'  推理: {((t2-t1)*1000):.0f}ms, 输出: {rendered.shape}, 坐标: {coords}')
else:
    print(f'  推理: {((t2-t1)*1000):.0f}ms (无人脸 — dummy帧正常)')
p.stop()
print('  ✓ 预热成功')
" || {
    echo "预热失败!"
    exit 1
}

echo ""
echo "============================================================"
echo "  部署完成!"
echo "============================================================"
echo ""
echo "  下一步:"
echo "    1. 上传视频文件到服务器"
echo "    2. 运行压测:"
echo "       python benchmark_local.py --video <视频文件> --size 256 --output result.mp4"
echo ""
echo "  如需 80 端口开放 TCP 服务端给外网:"
echo "    1. AutoDL 控制台 → 自定义服务 → 添加端口 17864"
echo "    2. 运行 python stream_server_tcp.py --size 256"
echo "    3. 本地客户端 windows_client_tcp.py 改 WSL_HOST 为服务器公网IP"
echo ""
