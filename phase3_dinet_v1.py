"""
Phase 3: DINetV1 直接加载 + 逐帧推理 ✅
完全脱离 .so，直接实例化 DINetV1 并加载权重
"""
import os, sys, time, numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"


class DINetRenderer:
    """DINet 渲染器封装，提供简洁推理接口"""

    def __init__(self, device="cuda"):
        self.device = device
        from landmark2face_wy.models.networks import DINetV1

        print("创建 DINetV1...")
        self.net = DINetV1(source_channel=3, ref_channel=3, audio_channel=256)

        # 加载权重
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        miss, unexp = self.net.load_state_dict(ckpt["face_G"], strict=False)
        print(f"  权重: {sum(p.numel() for p in self.net.parameters()):,} params, "
              f"missing={len(miss)}, unexpected={len(unexp)}")

        self.net.eval()
        self.net.to(device)

        print(f"  设备: {device}, 显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    def render(self, source_face, ref_face, audio_features):
        """
        逐帧渲染

        Args:
            source_face: [B, 3, 256, 256] tensor or numpy, 源人脸
            ref_face: [B, 3, 256, 256] tensor or numpy, 参考人脸
            audio_features: [B, 256, 256] tensor or numpy, 音频特征 (MFCC padded)

        Returns:
            rendered: [B, 3, 256, 256] tensor, 渲染后人脸
        """
        if isinstance(source_face, np.ndarray):
            source_face = torch.from_numpy(source_face).float()
        if isinstance(ref_face, np.ndarray):
            ref_face = torch.from_numpy(ref_face).float()
        if isinstance(audio_features, np.ndarray):
            audio_features = torch.from_numpy(audio_features).float()

        source_face = source_face.to(self.device)
        ref_face = ref_face.to(self.device)
        audio_features = audio_features.to(self.device)

        # 确保正确维度
        if source_face.dim() == 3:
            source_face = source_face.unsqueeze(0)
        if ref_face.dim() == 3:
            ref_face = ref_face.unsqueeze(0)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)

        with torch.no_grad():
            output = self.net(source_face, ref_face, audio_features)

        return output  # [B, 3, 256, 256]

    def benchmark(self, iterations=100):
        """推理速度测试"""
        src = torch.randn(1, 3, 256, 256).to(self.device)
        ref = torch.randn(1, 3, 256, 256).to(self.device)
        audio = torch.randn(1, 256, 256).to(self.device)

        # warmup
        for _ in range(10):
            self.net(src, ref, audio)
        torch.cuda.synchronize()

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.net(src, ref, audio)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        print(f"\n=== DINet 推理基准 ===")
        print(f"  平均: {times.mean():.1f}ms  ({1000/times.mean():.0f} FPS)")
        print(f"  中位: {np.median(times):.1f}ms")
        print(f"  最小: {times.min():.1f}ms / 最大: {times.max():.1f}ms")
        print(f"  显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        return times


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: DINetV1 直接推理 ✅")
    print("=" * 60)
    renderer = DINetRenderer()
    renderer.benchmark()
    print("\nDINet 模型已完全脱离 .so，可逐帧推理！")
