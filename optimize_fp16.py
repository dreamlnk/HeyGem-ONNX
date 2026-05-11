"""
FP16 优化测试: DINet + ONNX 模型半精度推理
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"


def test_dinet_fp16():
    """测试 DINetV1 在 FP32 vs FP16 vs AMP 下的性能"""
    from landmark2face_wy.models.networks import DINetV1

    print("=" * 60)
    print("DINetV1 FP16 优化对比")
    print("=" * 60)

    # 加载权重
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    src = torch.randn(1, 3, 256, 256).cuda()
    ref = torch.randn(1, 3, 256, 256).cuda()
    audio = torch.randn(1, 256, 256).cuda()
    src_fp16 = src.half()
    ref_fp16 = ref.half()
    audio_fp16 = audio.half()

    results = {}

    # --- FP32 baseline ---
    print("\n--- FP32 (baseline) ---")
    net_fp32 = DINetV1(3, 3, 256).cuda().eval()
    net_fp32.load_state_dict(ckpt["face_G"], strict=False)
    torch.cuda.synchronize()
    for _ in range(10):
        net_fp32(src, ref, audio)
    torch.cuda.synchronize()

    times_fp32 = []
    for _ in range(100):
        t0 = time.perf_counter()
        net_fp32(src, ref, audio)
        torch.cuda.synchronize()
        times_fp32.append((time.perf_counter() - t0) * 1000)
    results["FP32"] = np.array(times_fp32)
    mem_fp32 = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  平均: {results['FP32'].mean():.1f}ms  ({1000/results['FP32'].mean():.0f} FPS)")
    print(f"  显存峰值: {mem_fp32:.2f}GB")

    del net_fp32
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- Pure FP16 ---
    print("\n--- Pure FP16 ---")
    net_fp16 = DINetV1(3, 3, 256).half().cuda().eval()
    net_fp16.load_state_dict(ckpt["face_G"], strict=False)
    torch.cuda.synchronize()
    for _ in range(10):
        net_fp16(src_fp16, ref_fp16, audio_fp16)
    torch.cuda.synchronize()

    times_fp16 = []
    nan_count = 0
    for _ in range(100):
        t0 = time.perf_counter()
        out = net_fp16(src_fp16, ref_fp16, audio_fp16)
        torch.cuda.synchronize()
        times_fp16.append((time.perf_counter() - t0) * 1000)
        if torch.isnan(out).any() or torch.isinf(out).any():
            nan_count += 1
    results["Pure FP16"] = np.array(times_fp16)
    mem_fp16 = torch.cuda.max_memory_allocated() / 1024**3
    speedup = results["FP32"].mean() / results["Pure FP16"].mean()
    print(f"  平均: {results['Pure FP16'].mean():.1f}ms  ({1000/results['Pure FP16'].mean():.0f} FPS)")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  NaN/Inf: {nan_count}/100")
    print(f"  显存峰值: {mem_fp16:.2f}GB  (节省 {(1-mem_fp16/mem_fp32)*100:.0f}%)")

    del net_fp16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- AMP (Automatic Mixed Precision) ---
    print("\n--- AMP (混合精度) ---")
    net_amp = DINetV1(3, 3, 256).cuda().eval()
    net_amp.load_state_dict(ckpt["face_G"], strict=False)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # inference only
    torch.cuda.synchronize()
    for _ in range(10):
        with torch.cuda.amp.autocast():
            net_amp(src, ref, audio)
    torch.cuda.synchronize()

    times_amp = []
    nan_count = 0
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast():
            out = net_amp(src, ref, audio)
        torch.cuda.synchronize()
        times_amp.append((time.perf_counter() - t0) * 1000)
        if torch.isnan(out).any() or torch.isinf(out).any():
            nan_count += 1
    results["AMP"] = np.array(times_amp)
    speedup = results["FP32"].mean() / results["AMP"].mean()
    print(f"  平均: {results['AMP'].mean():.1f}ms  ({1000/results['AMP'].mean():.0f} FPS)")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  NaN/Inf: {nan_count}/100")

    del net_amp
    torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{'方法':<15} {'平均(ms)':<12} {'FPS':<10} {'加速比':<10} {'显存(GB)':<12}")
    print(f"{'-'*60}")
    print(f"{'FP32':<15} {results['FP32'].mean():<12.1f} {1000/results['FP32'].mean():<10.0f} {'1.0x':<10} {mem_fp32:<12.2f}")
    if "Pure FP16" in results:
        fp16_speedup = results["FP32"].mean() / results["Pure FP16"].mean()
        print(f"{'Pure FP16':<15} {results['Pure FP16'].mean():<12.1f} {1000/results['Pure FP16'].mean():<10.0f} {f'{fp16_speedup:.1f}x':<10} {mem_fp16:<12.2f}")
    if "AMP" in results:
        amp_speedup = results["FP32"].mean() / results["AMP"].mean()
        print(f"{'AMP':<15} {results['AMP'].mean():<12.1f} {1000/results['AMP'].mean():<10.0f} {f'{amp_speedup:.1f}x':<10} {'-':<12}")

    return results


def test_onnx_fp16():
    """测试 ONNX 模型在 FP16 下的性能"""
    import onnxruntime

    print(f"\n{'='*60}")
    print("ONNX 模型 FP16 测试")
    print("=" * 60)

    models = {
        "SCRFD": "face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx",
        "BiseNet": "pretrain_models/face_lib/face_parsing/79999_iter.onnx",
    }

    for name, path in models.items():
        print(f"\n--- {name} ---")
        if not os.path.exists(path):
            print(f"  模型不存在: {path}")
            continue

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # FP32
        try:
            s_fp32 = onnxruntime.InferenceSession(
                path, opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            init_time = time.perf_counter()
            input_info = s_fp32.get_inputs()[0]
            inp = np.random.randn(*input_info.shape).astype(np.float32)
            # warmup
            for _ in range(5):
                s_fp32.run(None, {input_info.name: inp})
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                s_fp32.run(None, {input_info.name: inp})
                times.append((time.perf_counter() - t0) * 1000)
            times = np.array(times)
            print(f"  FP32: {times.mean():.1f}ms  ({1000/times.mean():.0f} FPS)")

            # FP16 (ONNX supports FP16 mode)
            inp_fp16 = inp.astype(np.float16)
            try:
                s_fp16 = onnxruntime.InferenceSession(
                    path, opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                for _ in range(5):
                    s_fp16.run(None, {input_info.name: inp_fp16})
                times16 = []
                for _ in range(50):
                    t0 = time.perf_counter()
                    s_fp16.run(None, {input_info.name: inp_fp16})
                    times16.append((time.perf_counter() - t0) * 1000)
                times16 = np.array(times16)
                speedup = times.mean() / times16.mean()
                print(f"  FP16: {times16.mean():.1f}ms  ({1000/times16.mean():.0f} FPS) 加速比: {speedup:.1f}x")
            except Exception as e:
                print(f"  FP16: 不支持 ({e})")
        except Exception as e:
            print(f"  错误: {e}")


if __name__ == "__main__":
    test_dinet_fp16()
    test_onnx_fp16()
    print("\n优化测试完成")
