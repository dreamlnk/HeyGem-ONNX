"""
Phase 3: DINet 模型推理验证
从 .so 加载模型架构，导入权重，测试推理速度
"""
import os
import sys
import time
import numpy as np
import torch

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"


def load_model():
    """从 .so 加载 DINet 模型并导入权重"""
    import argparse
    from landmark2face_wy.options.test_options import TestOptions

    parser = argparse.ArgumentParser()
    opt = TestOptions()
    opt.initialize(parser)
    opt.parse()  # 解析默认参数

    # DINet 模型需要的参数覆盖
    opt.model = "l2faceaudio"
    opt.name = "anylang"
    opt.img_size = 256
    opt.audio_feature = "wenet"
    opt.feature_path = "./landmark2face_wy/feature"
    opt.batch_size = 1
    opt.gpu_ids = [0]

    print(f"Creating model: {opt.model}")
    from landmark2face_wy.models import create_model
    model = create_model(opt)

    # 加载权重
    import numpy
    import torch.serialization
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    model.netG.load_state_dict(ckpt["face_G"], strict=False)
    model.eval()
    model.to("cuda")

    print(f"Model loaded, params: {sum(p.numel() for p in model.netG.parameters()):,}")
    return model, opt


def benchmark(model, iterations=50):
    """推理速度测试"""
    print(f"\n--- 推理基准 ({iterations}次) ---")
    # 创建虚拟输入
    dummy_img = torch.randn(1, 3, 256, 256).cuda()
    dummy_audio = torch.randn(1, 1, 256, 256).cuda()
    dummy_mask = torch.ones(1, 1, 256, 256).cuda()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            try:
                _ = model.netG(dummy_img, dummy_img, dummy_audio, dummy_audio, dummy_mask, dummy_mask)
            except Exception as e:
                print(f"Warmup error: {e}")
                break

    # Benchmark
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                _ = model.netG(dummy_img, dummy_img, dummy_audio, dummy_audio, dummy_mask, dummy_mask)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                print(f"Inference error: {e}")
                break

    if times:
        times = np.array(times)
        print(f"  平均: {times.mean():.1f}ms  ({1000/times.mean():.0f} FPS)")
        print(f"  中位: {np.median(times):.1f}ms")
        print(f"  最小: {times.min():.1f}ms  /  最大: {times.max():.1f}ms")
    else:
        print("  No successful inference runs")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: DINet 模型推理验证")
    print("=" * 60)

    try:
        model, opt = load_model()
        benchmark(model)
        print("\nPhase 3 完成 ✓")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
