"""
torch.compile 优化测试: DINetV1
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"


def benchmark(model, src, ref, audio, label, iterations=100):
    # warmup
    for _ in range(10):
        model(src, ref, audio)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model(src, ref, audio)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    fps = 1000 / times.mean()
    print(f"  {label}: {times.mean():.1f}ms ({fps:.0f} FPS) "
          f"[min={times.min():.1f}, max={times.max():.1f}, p99={np.percentile(times,99):.1f}]")
    return times


def main():
    from landmark2face_wy.models.networks import DINetV1

    print("=" * 60)
    print("DINetV1 torch.compile 优化测试")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print("=" * 60)

    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    src = torch.randn(1, 3, 256, 256).cuda()
    ref = torch.randn(1, 3, 256, 256).cuda()
    audio = torch.randn(1, 256, 256).cuda()

    results = {}

    # --- Eager (baseline) ---
    print("\n--- Eager mode (baseline) ---")
    torch.cuda.reset_peak_memory_stats()
    model_eager = DINetV1(3, 3, 256).cuda().eval()
    model_eager.load_state_dict(ckpt["face_G"], strict=False)
    results["eager"] = benchmark(model_eager, src, ref, audio, "Eager")
    mem_eager = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  显存峰值: {mem_eager:.2f}GB")

    # --- torch.compile: reduce-overhead ---
    print("\n--- torch.compile (reduce-overhead) ---")
    torch.cuda.reset_peak_memory_stats()
    model_ro = DINetV1(3, 3, 256).cuda().eval()
    model_ro.load_state_dict(ckpt["face_G"], strict=False)
    model_ro = torch.compile(model_ro, mode="reduce-overhead")
    results["compile_ro"] = benchmark(model_ro, src, ref, audio, "compile(reduce-overhead)")
    mem_ro = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  显存峰值: {mem_ro:.2f}GB")

    # --- torch.compile: max-autotune ---
    print("\n--- torch.compile (max-autotune) ---")
    torch.cuda.reset_peak_memory_stats()
    model_ma = DINetV1(3, 3, 256).cuda().eval()
    model_ma.load_state_dict(ckpt["face_G"], strict=False)
    model_ma = torch.compile(model_ma, mode="max-autotune")
    results["compile_ma"] = benchmark(model_ma, src, ref, audio, "compile(max-autotune)", iterations=50)
    mem_ma = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  显存峰值: {mem_ma:.2f}GB")

    # --- torch.compile: default ---
    print("\n--- torch.compile (default) ---")
    torch.cuda.reset_peak_memory_stats()
    model_def = DINetV1(3, 3, 256).cuda().eval()
    model_def.load_state_dict(ckpt["face_G"], strict=False)
    model_def = torch.compile(model_def)
    results["compile_default"] = benchmark(model_def, src, ref, audio, "compile(default)")
    mem_def = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  显存峰值: {mem_def:.2f}GB")

    # --- Summary ---
    baseline_mean = results["eager"].mean()
    print(f"\n{'='*60}")
    print(f"{'方法':<28} {'平均(ms)':<12} {'FPS':<10} {'加速比':<10}")
    print(f"{'-'*60}")
    for name, times in results.items():
        label_map = {
            "eager": "Eager (baseline)",
            "compile_default": "compile (default)",
            "compile_ro": "compile (reduce-overhead)",
            "compile_ma": "compile (max-autotune)",
        }
        speedup = baseline_mean / times.mean()
        print(f"{label_map[name]:<28} {times.mean():<12.1f} {1000/times.mean():<10.0f} {f'{speedup:.2f}x':<10}")


if __name__ == "__main__":
    main()
