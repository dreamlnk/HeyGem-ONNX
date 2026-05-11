"""
CUDA Graph 优化测试: DINetV1
消除 kernel launch overhead，预期 5-15% 提升 (Pascal友好)
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"


def benchmark(fn, label, iterations=100):
    # warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    fps = 1000 / times.mean()
    print(f"  {label}: {times.mean():.1f}ms ({fps:.0f} FPS) "
          f"[min={times.min():.1f}, p99={np.percentile(times,99):.1f}]")
    return times


def main():
    from landmark2face_wy.models.networks import DINetV1

    print("=" * 60)
    print("DINetV1 CUDA Graph 优化测试")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print("=" * 60)

    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # --- Eager (baseline) ---
    print("\n--- Eager mode (baseline) ---")
    torch.cuda.reset_peak_memory_stats()
    model = DINetV1(3, 3, 256).cuda().eval()
    model.load_state_dict(ckpt["face_G"], strict=False)

    src = torch.randn(1, 3, 256, 256).cuda()
    ref = torch.randn(1, 3, 256, 256).cuda()
    audio = torch.randn(1, 256, 256).cuda()

    def eager_fn():
        with torch.no_grad():
            model(src, ref, audio)

    results_eager = benchmark(eager_fn, "Eager")
    mem_baseline = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  显存峰值: {mem_baseline:.2f}GB")

    # --- CUDA Graph ---
    print("\n--- CUDA Graph ---")

    # 创建独立输入 (graph 会修改它们)
    src_g = torch.randn(1, 3, 256, 256).cuda()
    ref_g = torch.randn(1, 3, 256, 256).cuda()
    audio_g = torch.randn(1, 256, 256).cuda()

    # Warmup for graph capture
    torch.cuda.synchronize()
    for _ in range(3):
        with torch.no_grad():
            model(src_g, ref_g, audio_g)
    torch.cuda.synchronize()

    # Capture graph
    torch.cuda.reset_peak_memory_stats()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            out_g = model(src_g, ref_g, audio_g)

    def graph_fn():
        # 每次更新静态输入 (必要，否则每次推理结果相同)
        src_g.copy_(src)
        ref_g.copy_(ref)
        audio_g.copy_(audio)
        g.replay()

    results_graph = benchmark(graph_fn, "CUDA Graph")
    mem_graph = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  显存峰值: {mem_graph:.2f}GB")

    # --- Summary ---
    baseline_mean = results_eager.mean()
    graph_mean = results_graph.mean()
    speedup = baseline_mean / graph_mean

    print(f"\n{'='*60}")
    print(f"{'方法':<20} {'平均(ms)':<12} {'FPS':<10} {'加速比':<10} {'显存':<10}")
    print(f"{'-'*60}")
    print(f"{'Eager':<20} {baseline_mean:<12.1f} {1000/baseline_mean:<10.0f} {'1.00x':<10} {mem_baseline:<10.2f}")
    print(f"{'CUDA Graph':<20} {graph_mean:<12.1f} {1000/graph_mean:<10.0f} {f'{speedup:.2f}x':<10} {mem_graph:<10.2f}")

    if speedup > 1.01:
        print(f"\nCUDA Graph 有效! 加速 {speedup:.2f}x, FPS 从 {1000/baseline_mean:.0f} → {1000/graph_mean:.0f}")


if __name__ == "__main__":
    main()
