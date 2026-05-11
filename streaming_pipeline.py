"""
混合流式管线 MVP
Python预处理(人脸检测/解析/音频) + .so DINet渲染 → 帧捕获输出
"""
import os
import sys
import time
import queue
import threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

# === 预处理模块 (纯Python, 无.so依赖) ===

from phase1_scrfd_test import load_session as scrfd_load, detect as scrfd_detect
from phase2_face_parsing import load_session as parse_load, parse_face as biseNet_parse
from phase2_audio_feature import extract_mfcc, prepare_dinet_input


class StreamingPipeline:
    def __init__(self, detect_interval=5):
        print("=" * 50)
        print("初始化混合流式管线...")

        # 加载ONNX模型
        print("  加载 SCRFD...")
        self.scrfd = scrfd_load()
        print("  加载 BiseNet...")
        self.parsing = parse_load()

        # GPU warmup
        print("  GPU warmup...")
        dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
        for _ in range(5):
            self.scrfd.run(None, {self.scrfd.get_inputs()[0].name: dummy})
        dummy_p = np.random.randn(1, 3, 512, 512).astype(np.float32)
        for _ in range(5):
            self.parsing.run(None, {self.parsing.get_inputs()[0].name: dummy_p})
        print("  Warmup 完成")

        # 音频缓冲 (最近3秒)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_lock = threading.Lock()

        # 帧输出队列
        self.output_queue = queue.Queue(maxsize=30)

        # 跳帧策略
        self.detect_interval = detect_interval
        self.frame_idx = 0

        # 状态
        self.running = False
        self.latest_face = None
        self.latest_kps = None
        self.latest_parsing = None
        self.latest_audio_feat = None
        self.last_face_center = None

        print("  初始化完成 ✓")
        print("=" * 50)

    def _face_moved(self, bbox, threshold=20):
        """检查人脸是否显著移动"""
        if self.last_face_center is None:
            return True
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        dist = abs(cx - self.last_face_center[0]) + abs(cy - self.last_face_center[1])
        return dist > threshold

    def process_frame(self, frame_bgr):
        """处理单帧: 人脸检测(跳帧) + 解析(按需)"""
        self.frame_idx += 1
        run_full = (self.frame_idx % self.detect_interval == 0)

        if run_full:
            bboxes, kpss, meta = scrfd_detect(self.scrfd, frame_bgr)

            if len(bboxes) > 0:
                face_moved = self._face_moved(bboxes[0])
                self.latest_face = bboxes[0]
                self.latest_kps = kpss[0] if len(kpss) > 0 else None
                self.last_face_center = (
                    (bboxes[0][0] + bboxes[0][2]) // 2,
                    (bboxes[0][1] + bboxes[0][3]) // 2,
                )

                # 只在人脸移动时重新解析
                if face_moved:
                    x1, y1, x2, y2 = bboxes[0]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    size = int(max(x2 - x1, y2 - y1) * 1.5)
                    h, w = frame_bgr.shape[:2]
                    fx1 = max(0, cx - size // 2)
                    fx2 = min(w, cx + size // 2)
                    fy1 = max(0, cy - size // 2)
                    fy2 = min(h, cy + size // 2)
                    face_crop = frame_bgr[fy1:fy2, fx1:fx2]
                    if face_crop.size > 0:
                        _, _, _ = biseNet_parse(self.parsing, face_crop)

        # 绘制检测框 (复用上一帧结果)
        if self.latest_face is not None:
            x1, y1, x2, y2 = self.latest_face
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.latest_kps is not None:
                kps = self.latest_kps.reshape(-1, 2)
                for kp in kps:
                    cv2.circle(frame_bgr, tuple(kp), 2, (0, 0, 255), -1)

        return frame_bgr

    def process_frame(self, frame_bgr):
        """处理单帧: 人脸检测 + 解析"""
        # 1. 人脸检测 (简化版: 每5帧检测一次以提速)
        bboxes, kpss, meta = scrfd_detect(self.scrfd, frame_bgr)

        if len(bboxes) > 0:
            self.latest_face = bboxes[0]
            self.latest_kps = kpss[0] if len(kpss) > 0 else None

            # 2. 裁切人脸区域
            x1, y1, x2, y2 = bboxes[0]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            size = int(max(x2 - x1, y2 - y1) * 1.5)
            h, w = frame_bgr.shape[:2]
            fx1 = max(0, cx - size // 2)
            fx2 = min(w, cx + size // 2)
            fy1 = max(0, cy - size // 2)
            fy2 = min(h, cy + size // 2)
            face_crop = frame_bgr[fy1:fy2, fx1:fx2]
            if face_crop.size > 0:
                parsing, _, _ = biseNet_parse(self.parsing, face_crop)
                self.latest_parsing = parsing

            # 绘制检测框
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.latest_kps is not None:
                kps = self.latest_kps.reshape(-1, 2)
                for kp in kps:
                    cv2.circle(frame_bgr, tuple(kp), 2, (0, 0, 255), -1)

        return frame_bgr

    def feed_audio(self, audio_chunk, sample_rate=16000):
        """喂入音频数据, 更新缓冲"""
        with self.audio_lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            # 保持最近3秒
            max_samples = sample_rate * 3
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

            # 提取MFCC特征
            if len(self.audio_buffer) >= sample_rate * 0.5:  # 至少0.5秒
                features = extract_mfcc(self.audio_buffer)
                self.latest_audio_feat = prepare_dinet_input(features)

    def start(self):
        """启动管线"""
        self.running = True
        print("管线已启动，等待输入...")

    def stop(self):
        """停止管线"""
        self.running = False
        print("管线已停止")


# === 测试: 摄像头实时预览 ===

def test_webcam():
    """测试摄像头捕获+人脸检测 (不经过DINet)"""
    print("打开摄像头...")
    cap = cv2.VideoCapture(0)  # WSL可能无法访问摄像头
    if not cap.isOpened():
        print("无法打开摄像头 (WSL环境)，使用视频文件测试")
        video_path = "/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/example/video.mp4"
        cap = cv2.VideoCapture(video_path)

    pipeline = StreamingPipeline()
    pipeline.start()

    frame_count = 0
    fps_start = time.time()

    while pipeline.running:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)
        frame_count += 1

        # 显示FPS
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"\rFPS: {fps:.1f}, 人脸: {pipeline.latest_face is not None}", end="")
            frame_count = 0
            fps_start = time.time()

        # 显示结果 (WSL无法显示窗口，仅统计)
        if frame_count % 30 == 0:
            pass  # 在WSL中无法显示X窗口

    cap.release()
    pipeline.stop()
    print(f"\n处理完成")


# === 单元测试: 完整管线一段 ===

def test_pipeline_segment():
    """测试完整预处理管线 (不含DINet渲染)"""
    import subprocess

    print("=" * 50)
    print("预处理管线基准测试")
    print("=" * 50)

    pipeline = StreamingPipeline()

    # 从示例视频提取一帧和音频
    video_path = "/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/example/video.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("无法读取视频")
        return

    # 提取音频
    audio_path = "/tmp/test_audio.wav"
    subprocess.run(
        f"ffmpeg -y -i '{video_path}' -ac 1 -ar 16000 -t 3 {audio_path}",
        shell=True, capture_output=True
    )

    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    pipeline.feed_audio(audio)

    # 测试处理速度
    print(f"\n测试图片: {frame.shape[1]}x{frame.shape[0]}")
    times = []
    for i in range(50):
        t0 = time.perf_counter()
        pipeline.process_frame(frame.copy())
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"预处理 (检测+解析):")
    print(f"  平均: {times.mean():.1f}ms  ({1000/times.mean():.0f} FPS)")
    print(f"  中位: {np.median(times):.1f}ms")
    print(f"  最小: {times.min():.1f}ms / 最大: {times.max():.1f}ms")
    print(f"\n管线状态:")
    print(f"  人脸检测: {'✓' if pipeline.latest_face is not None else '✗'}")
    print(f"  人脸解析: {'✓' if pipeline.latest_parsing is not None else '✗'}")
    print(f"  音频特征: {'✓' if pipeline.latest_audio_feat is not None else '✗'}")


if __name__ == "__main__":
    test_pipeline_segment()
