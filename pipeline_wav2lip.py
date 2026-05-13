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
    def __init__(self, detect_interval=3, test_audio=False):
        print("=" * 60)
        print("Wav2Lip Streaming Pipeline")
        print("=" * 60)

        print("  Loading YuNet...")
        self.detector = scrfd_load()

        print("  Loading Wav2Lip...")
        self.wav2lip = Wav2LipInferenceEngine()

        self.test_audio = test_audio
        self.detect_interval = detect_interval
        self.frame_idx = 0

        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_lock = threading.Lock()
        self.latest_mel = None

        # Face tracking
        self.last_bbox = None
        self.last_kps = None

        # Previous face for Wav2Lip 2-frame stacking (RGB, [-1,1], (3,96,96))
        self.prev_face_norm = None

        # CUDA inference lock
        self.wav2lip_lock = threading.Lock()

        # Debug
        self._last_rendered_96 = None

        # Output queue and state
        self.output_queue = queue.Queue(maxsize=30)
        self.running = False

        print("  Warming up...")
        self._warmup()
        print("  Ready ✓")
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
            return self._get_test_mel()
        with self.audio_lock:
            if len(self.audio_buffer) < 3200:  # < 200ms
                return np.zeros((1, 1, 80, 16), dtype=np.float32)
            mel = mel_spectrogram(self.audio_buffer)
        return get_wav2lip_mel_input(mel)

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
                self.last_bbox = None
                self.last_kps = None
                self.prev_face_norm = None

        if self.last_bbox is None:
            return frame_bgr

        if not self._valid_detection(self.last_bbox, self.last_kps, frame_bgr):
            self.last_bbox = None
            self.last_kps = None
            self.prev_face_norm = None
            return frame_bgr

        # --- 2. Crop face region from bbox ---
        x1, y1, x2, y2 = self.last_bbox.astype(int)
        bw, bh = x2 - x1, y2 - y1

        # Wav2Lip-style asymmetric padding: no left/right, slight top, larger bottom for chin
        pad_top = int(bh * 0.05)
        pad_bottom = int(bh * 0.15)
        pad_left = int(bw * 0.05)
        pad_right = int(bw * 0.05)

        cx1 = max(0, x1 - pad_left)
        cy1 = max(0, y1 - pad_top)
        cx2 = min(W, x2 + pad_right)
        cy2 = min(H, y2 + pad_bottom)

        if cx2 - cx1 < 10 or cy2 - cy1 < 10:
            return frame_bgr

        face_crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
        crop_h, crop_w = face_crop.shape[:2]

        # --- 3. Preprocess for Wav2Lip ---
        face_96 = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_AREA)
        face_stack, current_norm = preprocess_face_wav2lip(face_96, self.prev_face_norm)
        self.prev_face_norm = current_norm

        # --- 4. Get mel input ---
        if self.test_audio:
            self.feed_test_audio()
        mel_input = torch.from_numpy(self._get_mel_input())

        # --- 5. Wav2Lip inference ---
        with self.wav2lip_lock:
            rendered_96 = self.wav2lip.infer(face_stack.unsqueeze(0), mel_input)

        # --- 6. Composite back into original frame ---
        rendered_resized = cv2.resize(rendered_96, (crop_w, crop_h),
                                      interpolation=cv2.INTER_LINEAR)

        # Feather mask for smooth blending
        mask = _create_feather_mask(crop_h, crop_w)
        mask_3 = mask[..., np.newaxis]  # (h, w, 1)

        roi = frame_bgr[cy1:cy2, cx1:cx2].astype(np.float32)
        rendered_f = rendered_resized.astype(np.float32)
        blended = rendered_f * mask_3 + roi * (1.0 - mask_3)
        frame_bgr[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)

        # --- Debug output ---
        if self.frame_idx % 15 == 1:
            cv2.imwrite("/tmp/wav2lip_face_crop.png", face_crop)
            cv2.imwrite("/tmp/wav2lip_rendered_96.png", rendered_96)
            cv2.imwrite("/tmp/wav2lip_composited.png", frame_bgr)
            if self._last_rendered_96 is not None:
                diff = np.abs(rendered_96.astype(float) -
                              self._last_rendered_96.astype(float)).mean()
                mouth = rendered_96[48:, :, :]  # lower half
                mouth_prev = self._last_rendered_96[48:, :, :]
                mouth_diff = np.abs(mouth.astype(float) - mouth_prev.astype(float)).mean()
                print(f"\r[Wav2Lip] frame={self.frame_idx} "
                      f"face_diff={diff:.1f} mouth_diff={mouth_diff:.1f}",
                      end="", flush=True, file=sys.stderr)
            self._last_rendered_96 = rendered_96.copy()

        return frame_bgr

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        self.prev_face_norm = None
        self.last_bbox = None
        self.last_kps = None
