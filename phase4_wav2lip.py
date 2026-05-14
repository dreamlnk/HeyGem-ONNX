"""Wav2Lip inference engine — loads generator, pre-allocates CUDA tensors."""
import os
import numpy as np
import cv2
import torch

from models_wav2lip.wav2lip import Wav2Lip


class Wav2LipInferenceEngine:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base = os.path.join(os.path.dirname(__file__), "pretrain_models")
            # Prefer non-GAN checkpoint if available
            candidates = [
                os.path.join(base, "wav2lip_gan.pth"),
                os.path.join(base, "wav2lip.pth"),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

        self.device = torch.device("cuda")
        self.model = Wav2Lip().to(self.device).eval()

        ckpt = torch.load(model_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        # Strip "module." prefix if present (DataParallel wrapper)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.removeprefix("module."): v for k, v in state.items()}

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
            face_stack: (1, 6, 96, 96) float32 on CPU, in [0, 1]
            mel:        (1, 1, 80, 16) float32 on CPU, log-mel in [-4, 4]
        Returns:
            rendered: (96, 96, 3) uint8 BGR image
        """
        with torch.no_grad():
            self._face.copy_(face_stack)
            self._mel.copy_(mel)
            out = self.model(self._mel, self._face)  # (1, 3, 96, 96) [0, 1]

        out = out.float().cpu().numpy().squeeze(0)           # (3, 96, 96)
        out = np.clip(out, 0.0, 1.0)
        out = (out * 255).astype(np.uint8).transpose(1, 2, 0)  # (96, 96, 3) BGR
        return out


def preprocess_face_wav2lip(face_bgr_96):
    """Convert a 96x96 BGR face crop to Wav2Lip input tensor.

    Official convention: stack [lower-half-masked face, full face] as 6 BGR channels, /255.
    Model was trained with BGR (OpenCV native), NOT RGB.

    Args:
        face_bgr_96: (96, 96, 3) uint8 BGR face crop
    Returns:
        face_stack: (6, 96, 96) float32 tensor
    """
    # Zero out lower half (rows 48:96) — official approach
    face_masked = face_bgr_96.copy()
    face_masked[48:, :, :] = 0

    face_norm = face_bgr_96.astype(np.float32) / 255.0
    masked_norm = face_masked.astype(np.float32) / 255.0

    # (96, 96, 6) = [masked_lower_BGR, full_BGR]
    stacked = np.concatenate([masked_norm, face_norm], axis=-1)
    return torch.from_numpy(stacked).permute(2, 0, 1)  # (6, 96, 96)

