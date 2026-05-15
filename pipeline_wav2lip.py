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
# Canonical face landmarks for alignment (normalized 0-1, then scaled)
# YuNet 5-point order: left_eye, right_eye, nose, left_mouth, right_mouth
# ---------------------------------------------------------------------------

def _canonical_landmarks(size):
    """Return canonical 5-point landmarks at given image size."""
    pts = np.array([
        [0.35, 0.38],   # left eye
        [0.65, 0.38],   # right eye
        [0.50, 0.56],   # nose tip
        [0.35, 0.73],   # left mouth corner
        [0.65, 0.73],   # right mouth corner
    ], dtype=np.float32)
    return pts * size


def _compute_align_transform(src_pts, dst_pts):
    """Compute similarity transform (4-DOF: rotation, uniform scale, translation).
    Returns 2×3 affine matrix, or None if src_pts are degenerate."""
    if src_pts.shape != (5, 2) or dst_pts.shape != (5, 2):
        return None
    # Check for degenerate points (all same, collinear, etc.)
    src_std = src_pts.std(axis=0)
    if src_std[0] < 2 or src_std[1] < 2:
        return None
    try:
        M = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0)[0]
        return M  # None if estimation fails
    except cv2.error:
        return None


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
    def __init__(self, detect_interval=3, test_audio=False, size=96, use_align=None,
                 use_fp16=True):
        self.size = size
        # 5-point alignment too crude for any model — disabled by default
        if use_align is None:
            use_align = False
        self.use_align = use_align
        print("=" * 60)
        print(f"Wav2Lip Streaming Pipeline ({size}×{size})")
        print("=" * 60)

        print("  Loading YuNet...")
        self.detector = scrfd_load()

        print(f"  Loading Wav2Lip ({size}×{size})...")
        self.wav2lip = Wav2LipInferenceEngine(size=size, use_fp16=use_fp16)

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
        self._kps_history = []   # rolling buffer of keypoints for smoothing

        # Face alignment
        if self.use_align:
            self._canonical_lm = _canonical_landmarks(size)
            print(f"  Face alignment: ENABLED (similarity transform via 5-point landmarks)")

        # CUDA inference lock
        self.wav2lip_lock = threading.Lock()

        # Debug
        self._last_rendered_96 = None

        # Output queue and state
        self.output_queue = queue.Queue(maxsize=30)
        self.running = False

        with open("debug_pipeline.txt", "a") as df:
            df.write(f"=== SERVER START v4 size={size} fp16={'auto' if use_fp16 else False} ===\n")
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

    def _get_mel_input(self, t_detect=0, t_crop=0, t_prep=0, t_infer=0, t_total=0):
        if self.test_audio:
            self.feed_test_audio()
            idx = (self.frame_idx // 15) % 2
            return self._test_mels[idx]
        with self.audio_lock:
            buf_len = len(self.audio_buffer)
            if buf_len < 3200:  # < 200ms
                return np.zeros((1, 1, 80, 16), dtype=np.float32)
            # Only use last 0.5s (8000 samples) for mel — we only need last 16 frames
            # which correspond to ~0.2s of audio. Full 3s buffer is wasted work.
            audio_segment = self.audio_buffer[-8000:]
            buf_len = len(self.audio_buffer)
        mel = mel_spectrogram(audio_segment)
        mel_input = get_wav2lip_mel_input(mel)
        if self.frame_idx <= 5 or self.frame_idx % 30 == 0:
            with open("debug_pipeline.txt", "a") as df:
                df.write(f"F{self.frame_idx} AUDIO buf={buf_len} mel={mel_input.mean():.2f} "
                         f"detect={t_detect:.0f}ms crop={t_crop:.1f}ms prep={t_prep:.1f}ms "
                         f"infer={t_infer:.0f}ms total={t_total:.0f}ms\n")
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
        t0 = time.perf_counter()
        self.frame_idx += 1
        H, W = frame_bgr.shape[:2]

        # --- 1. Face detection (skip frames) ---
        t_detect = 0.0
        do_detect = (self.frame_idx % self.detect_interval == 0) or (self.last_bbox is None)
        if do_detect:
            t_d0 = time.perf_counter()
            # Detector at fixed long edge — preserve aspect ratio to avoid distortion
            det_long = 480
            if W >= H:
                det_w, det_h = det_long, int(H * det_long / W)
            else:
                det_h, det_w = det_long, int(W * det_long / H)
            det_w, det_h = max(det_w, 32), max(det_h, 32)
            det_frame = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
            bboxes, kpss, _ = scrfd_detect(self.detector, det_frame)
            # Scale bboxes/keypoints back to original frame coordinates
            sx, sy = W / det_w, H / det_h
            bboxes = [np.array([b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]) for b in bboxes]
            kpss = [k * np.array([sx, sy, sx, sy, sx, sy, sx, sy, sx, sy]) for k in kpss]
            t_detect = (time.perf_counter() - t_d0) * 1000
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
                self._kps_history.clear()

        if self.last_bbox is None:
            return None, None

        if not self._valid_detection(self.last_bbox, self.last_kps, frame_bgr):
            self.last_bbox = None
            self.last_kps = None
            self._bbox_history.clear()
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
        crop_sz = max(bw, bh) * 1.1
        crop_half = crop_sz / 2.0
        cx1 = int(cx - crop_half)
        cy1 = int(cy - crop_half)
        cx2 = int(cx + crop_half)
        cy2 = int(cy + crop_half)

        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(W, cx2)
        cy2 = min(H, cy2)

        if cx2 - cx1 < 10 or cy2 - cy1 < 10:
            return None, None

        t_crop0 = time.perf_counter()
        face_crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
        t_crop = (time.perf_counter() - t_crop0) * 1000
        crop_h, crop_w = face_crop.shape[:2]

        # --- 3. Face alignment (only for 256+ models) ---
        M_align = None
        if self.use_align and self.last_kps is not None:
            # Convert keypoints to crop-relative coordinates
            kps_img = self.last_kps.reshape(-1, 2).astype(np.float32)
            kps_crop = kps_img.copy()
            kps_crop[:, 0] -= cx1
            kps_crop[:, 1] -= cy1

            # Smooth keypoints
            self._kps_history.append(kps_crop.copy())
            if len(self._kps_history) > 5:
                self._kps_history.pop(0)
            kps_smooth = np.mean(self._kps_history, axis=0)

            # Compute alignment transform: crop coords → canonical coords
            M_align = _compute_align_transform(kps_smooth, self._canonical_lm)

        # --- 4. Preprocess for Wav2Lip ---
        if M_align is not None:
            # Warp face crop to canonical position (aligned)
            face_aligned = cv2.warpAffine(face_crop, M_align, (self.size, self.size),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE)
            # Compute inverse transform to restore original position after inference
            M_inv = cv2.invertAffineTransform(M_align)
        else:
            # Fallback: simple resize (no alignment)
            face_aligned = cv2.resize(face_crop, (self.size, self.size),
                                       interpolation=cv2.INTER_AREA)
            M_inv = None

        t_prep0 = time.perf_counter()
        face_stack = preprocess_face_wav2lip(face_aligned, size=self.size)
        t_prep = (time.perf_counter() - t_prep0) * 1000

        # --- 5. Get mel input ---
        t_mel0 = time.perf_counter()
        mel_input = torch.from_numpy(self._get_mel_input(
            t_detect=t_detect, t_crop=t_crop, t_prep=t_prep,
            t_infer=0, t_total=(time.perf_counter() - t0) * 1000))
        t_mel = (time.perf_counter() - t_mel0) * 1000

        # --- 6. Wav2Lip inference ---
        t_infer0 = time.perf_counter()
        with self.wav2lip_lock:
            rendered = self.wav2lip.infer(face_stack.unsqueeze(0), mel_input)
        t_infer = (time.perf_counter() - t_infer0) * 1000

        # --- 7. Restore original face position ---
        t_post0 = time.perf_counter()
        if M_inv is not None:
            rendered_unaligned = cv2.warpAffine(rendered, M_inv, (crop_w, crop_h),
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REPLICATE)
            rendered = cv2.resize(rendered_unaligned, (self.size, self.size),
                                    interpolation=cv2.INTER_AREA)
        t_post = (time.perf_counter() - t_post0) * 1000

        # Profile logging every 30 frames
        t_total = (time.perf_counter() - t0) * 1000
        if self.frame_idx <= 10 or self.frame_idx % 30 == 0:
            with open("debug_pipeline.txt", "a") as df:
                df.write(f"F{self.frame_idx} TIMING detect={t_detect:.0f}ms crop={t_crop:.1f}ms "
                         f"prep={t_prep:.1f}ms mel={t_mel:.1f}ms infer={t_infer:.0f}ms "
                         f"post={t_post:.1f}ms total={t_total:.0f}ms\n")

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
