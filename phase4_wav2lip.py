"""Wav2Lip inference engine — loads generator, pre-allocates CUDA tensors."""
import os
import numpy as np
import cv2
import torch

from models_wav2lip.wav2lip import Wav2Lip


class Wav2LipInferenceEngine:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "pretrain_models", "wav2lip_gan.pth"
            )

        self.device = torch.device("cuda")
        self.model = Wav2Lip().to(self.device).eval()

        ckpt = torch.load(model_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state)
        print(f"[Wav2Lip] Loaded checkpoint from {model_path}")

        # Pre-allocate static tensors on GPU
        self._face = torch.empty(1, 6, 96, 96, device=self.device)
        self._mel = torch.empty(1, 1, 80, 16, device=self.device)

    def warmup(self, n: int = 3):
        """Run dummy forward passes to warm CUDA kernels."""
        dummy_face = torch.randn(1, 6, 96, 96, device=self.device)
        dummy_mel = torch.randn(1, 1, 80, 16, device=self.device)
        for _ in range(n):
            self.model(dummy_mel, dummy_face)
        torch.cuda.synchronize()
        print("[Wav2Lip] Warmup complete")

    def infer(self, face_stack: torch.Tensor, mel: torch.Tensor) -> np.ndarray:
        """Run Wav2Lip inference for a single frame.

        Args:
            face_stack: (1, 6, 96, 96) float32 on CPU, 2-frame RGB stack in [-1, 1]
            mel:        (1, 1, 80, 16) float32 on CPU, log-mel
        Returns:
            rendered: (96, 96, 3) uint8 BGR image
        """
        with torch.no_grad():
            self._face.copy_(face_stack)
            self._mel.copy_(mel)
            out = self.model(self._mel, self._face)  # (1, 3, 96, 96) RGB [0, 1]

        out = out.float().cpu().numpy().squeeze(0)           # (3, 96, 96)
        out = np.clip(out, 0.0, 1.0)
        out = (out * 255).astype(np.uint8).transpose(1, 2, 0)  # (96, 96, 3) RGB
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out


def preprocess_face_wav2lip(face_bgr_96, prev_face_rgb_norm=None):
    """Convert a 96x96 BGR face crop to Wav2Lip input tensors.

    Args:
        face_bgr_96: (96, 96, 3) uint8 BGR face crop
        prev_face_rgb_norm: (3, 96, 96) float32 [-1,1] from previous frame, or None
    Returns:
        face_stack: (6, 96, 96) float32 tensor
        current_rgb_norm: (3, 96, 96) float32, to pass as prev_face next frame
    """
    face_rgb = cv2.cvtColor(face_bgr_96, cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) / 127.5) - 1.0  # -> [-1, 1]
    current = torch.from_numpy(face_norm).permute(2, 0, 1)     # (3, 96, 96)

    if prev_face_rgb_norm is None:
        prev = current
    else:
        prev = torch.from_numpy(prev_face_rgb_norm)

    stacked = torch.cat([current, prev], dim=0)  # (6, 96, 96)
    return stacked, face_norm.transpose(2, 0, 1)  # return CHW for next frame
