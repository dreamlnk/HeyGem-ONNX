"""Wav2Lip streaming pipeline — replaces DINet pipeline completely.

Reuses:
  - phase1_scrfd_test.py: YuNet face detection (unchanged)
  - stream_server_tcp.py: TCP server (1-line import change)
  - windows_client_tcp.py: entirely unchanged
"""
import sys
import os
import threading
import queue
import time
import numpy as np
import cv2
import torch

torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.path.dirname(__file__))

from phase1_scrfd_test import load_session as scrfd_load, detect as scrfd_detect
from phase2_audio_wav2lip import mel_spectrogram, get_wav2lip_mel_input
from phase4_wav2lip import Wav2LipInferenceEngine, preprocess_face_wav2lip


# ---------------------------------------------------------------------------
# Compositing helpers
# ---------------------------------------------------------------------------

def _create_feather_mask(h, w, feather_ratio=0.18):
    """Soft elliptical mask for seamless blending at crop boundaries.

    Returns (h, w) float32 mask, 1.0 in center, fading to 0 at edges.
    """
    cy, cx = h / 2.0, w / 2.0
    rx, ry = w / 2.0, h / 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)

    feather = max(1, int(min(h, w) * feather_ratio))
    inner = np.clip(1.0 - dist, 0, 1)
    mask = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), sigmaX=feather / 2.0)
    return np.clip(mask, 0, 1)


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    def __init__(self, detect_interval=3, test_audio=False, size=96):
        self.size = size
        print("=" * 60)
        print(f"Wav2Lip Streaming Pipeline ({size}×{size})")
        print("=" * 60)

        print("  Loading YuNet...")
        self.detector = scrfd_load()

        print(f"  Loading Wav2Lip ({size}×{size})...")
        self.wav2lip = Wav2LipInferenceEngine(size=size)

        self.test_audio = test_audio
        self.detect_interval = detect_interval
        self.frame_idx = 0

        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_lock = threading.Lock()
        self.latest_mel = None

        # Face tracking with temporal smoothing
        self.last_bbox = None
        self.last_kps = None
        self._bbox_history = []  # rolling buffer of (x1,y1,x2,y2) for smoothing

        # (No prev_face needed — Wav2Lip uses [masked_lower, full_face] stacking, not temporal)

        # CUDA inference lock
        self.wav2lip_lock = threading.Lock()

        # Debug
        self._last_rendered_96 = None

        # Output queue and state
        self.output_queue = queue.Queue(maxsize=30)
        self.running = False

        print("  Warming up...")
        self._warmup()
        print("  Ready")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self):
        # YuNet
        dummy = np.ones((640, 640, 3), dtype=np.uint8) * 128
        for _ in range(3):
            self.detector.setInputSize((640, 640))
            self.detector.detect(dummy)

        # Mel
        dummy_audio = np.zeros(16000, dtype=np.float32)
        _ = mel_spectrogram(dummy_audio)

        # Wav2Lip
        self.wav2lip.warmup(n=3)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def feed_audio(self, audio_chunk, sample_rate=16000):
        if self.test_audio:
            return
        with self.audio_lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            max_samples = sample_rate * 3  # 3-second rolling window
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

    def _get_mel_input(self):
        if self.test_audio:
            self.feed_test_audio()
            idx = (self.frame_idx // 15) % 2
            return self._test_mels[idx]
        with self.audio_lock:
            buf_len = len(self.audio_buffer)
            if buf_len < 3200:  # < 200ms
                return np.zeros((1, 1, 80, 16), dtype=np.float32)
            mel = mel_spectrogram(self.audio_buffer)
        mel_input = get_wav2lip_mel_input(mel)
        if self.frame_idx <= 5 or self.frame_idx % 30 == 0:
            with open("debug_pipeline.txt", "a") as df:
                df.write(f"Audio F{self.frame_idx}: buf={buf_len} mel_range=[{mel.min():.2f},{mel.max():.2f}] mel_mean={mel_input.mean():.3f}\n")
        return mel_input

    def feed_test_audio(self):
        if not hasattr(self, '_test_mels'):
            sr = 16000
            dur = 3.0
            samples = int(sr * dur)
            tt = np.linspace(0, dur, samples, endpoint=False, dtype=np.float32)

            freq = 100.0 + 700.0 * (tt / dur)
            sweep = np.sin(2.0 * np.pi * freq * tt).astype(np.float32) * 0.8
            silence = np.zeros(samples, dtype=np.float32)

            self._test_mels = [
                get_wav2lip_mel_input(mel_spectrogram(sweep)),
                get_wav2lip_mel_input(mel_spectrogram(silence)),
            ]
            print("  Test mels ready", flush=True, file=sys.stderr)

        idx = (self.frame_idx // 15) % 2
        return self._test_mels[idx]

    # ------------------------------------------------------------------
    # Detection (reused from pipeline_complete)
    # ------------------------------------------------------------------

    def _valid_detection(self, bbox, kps, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if x2 < 1 or x1 > w - 1 or y2 < 1 or y1 > h - 1:
            return False
        if bw < 20 or bh < 20 or bw > w * 1.2 or bh > h * 1.2:
            return False
        if bw <= 0 or bh <= 0:
            return False
        if kps is not None:
            if kps.ndim == 1:
                kps = kps.reshape(-1, 2)
            for kp in kps:
                if not (-w < kp[0] < 2 * w and -h < kp[1] < 2 * h):
                    return False
        return True

    def _has_skin(self, frame_bgr, bbox):
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox[:4]]
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return False
        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, (0, 15, 50), (25, 180, 255))
        ratio = skin.sum() / 255 / (roi.shape[0] * roi.shape[1])
        return ratio > 0.05

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr, profile=False):
        """Process a single BGR frame through the Wav2Lip pipeline."""
        self.frame_idx += 1
        H, W = frame_bgr.shape[:2]
        if self.frame_idx <= 3:
            with open("debug_conn.log", "a") as f:
                f.write(f"process_frame {self.frame_idx} {W}x{H}\n")

        # --- 1. Face detection (skip frames) ---
        do_detect = (self.frame_idx % self.detect_interval == 0) or (self.last_bbox is None)
        if do_detect:
            bboxes, kpss, _ = scrfd_detect(self.detector, frame_bgr)
            valid_found = False
            for i, bbox in enumerate(bboxes):
                kps_i = kpss[i] if i < len(kpss) else None
                if self._valid_detection(bbox, kps_i, frame_bgr) and self._has_skin(frame_bgr, bbox):
                    self.last_bbox = bbox
                    self.last_kps = kps_i
                    valid_found = True
                    break
            if not valid_found:
                if self.frame_idx <= 5 or self.frame_idx % 100 == 0:
                    with open("debug_pipeline.txt", "a") as df:
                        df.write(f"F{self.frame_idx} DETECT FAIL n={len(bboxes)}\n")
                self.last_bbox = None
                self.last_kps = None
                self._bbox_history.clear()
                if hasattr(self, '_kps_history'):
                    self._kps_history.clear()

        if self.last_bbox is None:
            return None, None

        if not self._valid_detection(self.last_bbox, self.last_kps, frame_bgr):
            self.last_bbox = None
            self.last_kps = None
            self._bbox_history.clear()
            if hasattr(self, '_kps_history'):
                self._kps_history.clear()
            return None, None

        # --- 2. Crop face region (bbox center, square, symmetric padding) ---
        raw_bbox = self.last_bbox.astype(float)
        self._bbox_history.append(raw_bbox.copy())
        if len(self._bbox_history) > 5:
            self._bbox_history.pop(0)
        smooth_bbox = np.mean(self._bbox_history, axis=0)
        x1, y1, x2, y2 = smooth_bbox
        bw, bh = x2 - x1, y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0 - bh * 0.05  # slight upward shift
        size = max(bw, bh) * 1.1
        half = size / 2.0
        cx1 = int(cx - half)
        cy1 = int(cy - half)
        cx2 = int(cx + half)
        cy2 = int(cy + half)

        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(W, cx2)
        cy2 = min(H, cy2)

        if cx2 - cx1 < 10 or cy2 - cy1 < 10:
            return None, None

        face_crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
        crop_h, crop_w = face_crop.shape[:2]

        # --- 3. Preprocess for Wav2Lip ---
        face_resized = cv2.resize(face_crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        face_stack = preprocess_face_wav2lip(face_resized, size=self.size)

        # --- 4. Get mel input ---
        mel_input = torch.from_numpy(self._get_mel_input())

        # --- 5. Wav2Lip inference ---
        with self.wav2lip_lock:
            rendered = self.wav2lip.infer(face_stack.unsqueeze(0), mel_input)

        # Return rendered face + crop coords.
        # Client computes delta = rendered - original_downscaled locally (no clipping).
        half = self.size // 2
        if self.frame_idx == 1:
            cv2.imwrite("debug_face_crop.png", face_crop)
            cv2.imwrite("debug_rendered_f1.png", rendered)
            with open("debug_pipeline.txt", "a") as df:
                df.write(f"F1 size={self.size} crop={crop_w}x{crop_h} rect=({cx1},{cy1})-({cx2},{cy2})\n")
        if self.frame_idx == 30:
            cv2.imwrite("debug_rendered_f30.png", rendered)
            with open("debug_pipeline.txt", "a") as df:
                df.write(f"F30 size={self.size} crop={crop_w}x{crop_h} rect=({cx1},{cy1})-({cx2},{cy2})\n")
                if os.path.exists("debug_rendered_f1.png"):
                    f1 = cv2.imread("debug_rendered_f1.png")
                    if f1 is not None:
                        diff = np.abs(f1.astype(float) - rendered.astype(float))
                        df.write(f"F1vsF30 diff: mean={diff.mean():.1f}/255 max={diff.max():.0f}/255 mouth_mean={diff[half:,:,:].mean():.1f}/255\n")
        if self._last_rendered_96 is not None and self.frame_idx <= 5:
            diff = np.abs(rendered.astype(float) -
                          self._last_rendered_96.astype(float)).mean()
            mouth = rendered[half:, :, :]
            mouth_prev = self._last_rendered_96[half:, :, :]
            mouth_diff = np.abs(mouth.astype(float) - mouth_prev.astype(float)).mean()
            with open("debug_pipeline.txt", "a") as df:
                df.write(f"F{self.frame_idx} df={diff:.1f} mouth_df={mouth_diff:.1f}\n")
        self._last_rendered_96 = rendered.copy()

        return rendered, (cx1, cy1, cx2, cy2)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        self.last_bbox = None
        self.last_kps = None
        self._bbox_history.clear()
        if hasattr(self, '_kps_history'):
            self._kps_history.clear()
