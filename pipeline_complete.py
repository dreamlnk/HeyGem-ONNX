"""
完整流式管线: 摄像头 → 人脸检测 → 对齐 → DINet渲染 → 合成输出
纯Python实现，不依赖任何.so文件
"""
import os, sys, time, queue, threading
import numpy as np
import cv2
import torch
import librosa

sys.path.insert(0, os.path.dirname(__file__))

from phase1_scrfd_test import load_session as scrfd_load, detect as scrfd_detect
from phase2_audio_feature import extract_mfcc, prepare_dinet_input

CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"

# === 256x256 标准人脸参考点 (ArcFace canonical × 256/112) ===
REFERENCE_POINTS = np.array([
    [87.53, 118.16],   # left eye
    [168.07, 118.16],  # right eye
    [128.06, 163.97],  # nose
    [94.97, 211.13],   # left mouth corner
    [161.67, 211.13],  # right mouth corner
], dtype=np.float32)


def estimate_similarity_transform(src_pts, dst_pts):
    """从5对关键点估算相似变换矩阵 (旋转+缩放+平移, 无剪切)"""
    # cv2.estimateAffinePartial2D 估算的就是相似变换
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    return M


def align_face(img_bgr, kps, target_size=256, ref_pts=REFERENCE_POINTS):
    """
    用5点关键点做相似变换对齐到标准位置
    Args:
        img_bgr: 原始BGR图像
        kps: [5, 2] 或 [10] 关键点 (SCRFD格式: 左眼,右眼,鼻尖,左嘴角,右嘴角)
        target_size: 输出尺寸
        ref_pts: 标准参考点
    Returns:
        aligned: [target_size, target_size, 3] BGR对齐人脸
        M: 2x3 仿射矩阵
    """
    if kps.ndim == 1:
        kps = kps.reshape(-1, 2)
    M = estimate_similarity_transform(kps.astype(np.float32), ref_pts.astype(np.float32))
    if M is None:
        return None, None
    aligned = cv2.warpAffine(img_bgr, M, (target_size, target_size),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned, M


def inverse_affine_transform(img_256, original_frame, M, mouth_mask=None):
    """
    将DINet输出(256x256)逆变换贴回原始帧 (仅处理人脸ROI区域)
    """
    h, w = original_frame.shape[:2]
    M_inv = cv2.invertAffineTransform(M)

    # 计算256x256四个角映射回原始帧的位置，确定ROI
    corners_256 = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype=np.float32)
    corners_orig = cv2.transform(corners_256.reshape(1, -1, 2), M_inv).reshape(-1, 2)

    x_min = max(0, int(corners_orig[:, 0].min()))
    x_max = min(w, int(corners_orig[:, 0].max()) + 1)
    y_min = max(0, int(corners_orig[:, 1].min()))
    y_max = min(h, int(corners_orig[:, 1].max()) + 1)
    roi_w, roi_h = x_max - x_min, y_max - y_min

    if roi_w <= 0 or roi_h <= 0:
        return original_frame

    # 调整M_inv只作用于ROI区域
    M_roi = M_inv.copy()
    M_roi[0, 2] -= x_min
    M_roi[1, 2] -= y_min

    # 只在ROI内做warp
    warped_roi = cv2.warpAffine(img_256, M_roi, (roi_w, roi_h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    result = original_frame.copy()
    roi = result[y_min:y_max, x_min:x_max]

    # 只覆盖warped中有内容的部分
    valid = (warped_roi.sum(axis=2) > 0)
    roi[valid] = warped_roi[valid]
    result[y_min:y_max, x_min:x_max] = roi

    return result


def create_eye_mask(size=256):
    """创建眼部遮罩 (训练时遮眼睛区域，推理时取消)"""
    mask = np.ones((size, size), dtype=np.uint8) * 255
    mask[20:70, 55:-55] = 0
    return mask


def create_mouth_mask(size=256):
    """创建嘴部遮罩 mask_B 风格 (下半脸区域)"""
    mask = np.ones((size, size), dtype=np.uint8) * 255
    half = size // 2  # 128
    mask[half - 45:246, 30:-30] = 0
    return mask


def preprocess_face_for_dinet(aligned_bgr, apply_eye_mask=True):
    """
    对齐人脸 → DINet输入格式
    Args:
        aligned_bgr: [256, 256, 3] BGR对齐人脸
        apply_eye_mask: 是否遮眼睛 (source/ref需要, mask_B不需要)
    Returns:
        tensor: [3, 256, 256] 归一化到[-1, 1]
    """
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    if apply_eye_mask:
        eye_mask = create_eye_mask()
        img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=eye_mask)
    # 归一化到 [-1, 1]: (x/255 - 0.5) / 0.5
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


def preprocess_mask_B_for_dinet(aligned_bgr):
    """
    创建mask_B: 对齐人脸 + 嘴部遮罩 (DINet输入中的参考遮罩)
    """
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    mouth_mask = create_mouth_mask()
    # 嘴部区域置0
    img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mouth_mask)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


def denormalize_output(tensor):
    """
    DINet输出 [-1, 1] → uint8 BGR
    Args:
        tensor: [1, 3, 256, 256] 或 [3, 256, 256]
    Returns:
        img_bgr: [256, 256, 3] uint8
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.float().clamp(-1, 1).cpu().numpy()
    img = img.transpose(1, 2, 0)  # CHW → HWC
    img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


class DINetInferenceEngine:
    """DINet推理引擎，管理模型生命周期和推理"""

    def __init__(self):
        from landmark2face_wy.models.networks import DINetV1

        print("  加载 DINetV1...")
        self.model = DINetV1(source_channel=3, ref_channel=3, audio_channel=256)
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["face_G"], strict=False)
        self.model.eval().cuda()
        print(f"  DINetV1 已加载, 显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

        # 预分配静态输入
        self.src = torch.zeros(1, 3, 256, 256, device="cuda")
        self.ref = torch.zeros(1, 3, 256, 256, device="cuda")
        self.audio = torch.zeros(1, 256, 256, device="cuda")

    def infer(self, source_face, ref_face, audio_features):
        """
        Args:
            source_face: [3, 256, 256] float32 tensor in [-1, 1]
            ref_face: [3, 256, 256] float32 tensor in [-1, 1]
            audio_features: [256, 256] float32 numpy
        Returns:
            rendered: [256, 256, 3] uint8 BGR
        """
        self.src.copy_(source_face.unsqueeze(0))
        self.ref.copy_(ref_face.unsqueeze(0))
        audio_t = torch.from_numpy(audio_features).float().unsqueeze(0).cuda()
        self.audio.copy_(audio_t)

        with torch.no_grad():
            output = self.model(self.src, self.ref, self.audio)

        return denormalize_output(output)


class StreamingPipeline:
    """完整流式管线"""

    def __init__(self, detect_interval=3):
        print("=" * 60)
        print("初始化完整流式管线")
        print("=" * 60)

        # 预处理模型
        print("  加载 SCRFD...")
        self.scrfd = scrfd_load()

        # DINet引擎
        self.dinet = DINetInferenceEngine()

        # GPU warmup
        print("  GPU warmup...")
        self._warmup()

        # 音频缓冲
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_lock = threading.Lock()
        self.latest_audio_feat = None

        # 人脸状态
        self.source_face_tensor = None  # 身份参考帧 (固定)
        self.last_bbox = None
        self.last_kps = None
        self.last_M = None  # 最近的对齐矩阵
        self.frame_idx = 0
        self.detect_interval = detect_interval

        # 输出队列
        self.output_queue = queue.Queue(maxsize=30)
        self.running = False

        print("  初始化完成 ✓")
        print(f"  跳帧: 每{detect_interval}帧检测一次")
        print("=" * 60)

    def _warmup(self):
        """GPU warmup消除首次推理延迟"""
        for _ in range(3):
            self.scrfd.run(None, {self.scrfd.get_inputs()[0].name:
                            np.random.randn(1, 3, 640, 640).astype(np.float32)})
        src = torch.randn(1, 3, 256, 256).cuda()
        ref = torch.randn(1, 3, 256, 256).cuda()
        audio = torch.randn(1, 256, 256).cuda()
        for _ in range(3):
            self.dinet.model(src, ref, audio)
        torch.cuda.synchronize()

    def set_source_face(self, frame_bgr, bbox, kps):
        """设置身份参考帧 (调用一次，后续复用)"""
        aligned, M = align_face(frame_bgr, kps)
        if aligned is not None:
            self.source_face_tensor = preprocess_face_for_dinet(aligned, apply_eye_mask=True)
            print(f"  身份参考帧已设置 (bbox={bbox})")

    def process_frame(self, frame_bgr, profile=False):
        """处理单帧"""
        self.frame_idx += 1
        t0 = time.perf_counter() if profile else 0

        # 1. 人脸检测 (跳帧)
        do_detect = (self.frame_idx % self.detect_interval == 0) or (self.last_bbox is None)
        if do_detect:
            bboxes, kpss, meta = scrfd_detect(self.scrfd, frame_bgr)
            if len(bboxes) > 0:
                self.last_bbox = bboxes[0]
                self.last_kps = kpss[0] if len(kpss) > 0 else None

        if self.last_bbox is None:
            return frame_bgr

        # 2. 人脸对齐 (复用关键点直到下次检测)
        if self.last_kps is not None:
            aligned_face, M = align_face(frame_bgr, self.last_kps)
            if aligned_face is not None:
                self.last_M = M
            else:
                return frame_bgr
        else:
            return frame_bgr

        # 3. 设置身份参考帧 (第一帧)
        if self.source_face_tensor is None and self.last_kps is not None:
            self.set_source_face(frame_bgr, self.last_bbox, self.last_kps)

        # 4. 准备DINet输入
        if self.source_face_tensor is None or self.latest_audio_feat is None:
            return self._draw_debug(frame_bgr)

        ref_tensor = preprocess_face_for_dinet(aligned_face, apply_eye_mask=True)

        # 5. DINet推理
        rendered_256 = self.dinet.infer(
            self.source_face_tensor,
            ref_tensor,
            self.latest_audio_feat,
        )

        # 6. 合成回原始帧
        result = inverse_affine_transform(rendered_256, frame_bgr, self.last_M)

        # 7. 画检测框 (调试用)
        result = self._draw_debug(result)

        if profile and self.frame_idx % 30 == 0:
            total = (time.perf_counter() - t0) * 1000
            print(f"  [Profile #{self.frame_idx}] total={total:.0f}ms")

        return result

    def _draw_debug(self, frame):
        """画检测框和关键点"""
        if self.last_bbox is not None:
            x1, y1, x2, y2 = self.last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.last_kps is not None:
                kps = self.last_kps.reshape(-1, 2)
                for kp in kps:
                    cv2.circle(frame, tuple(kp.astype(int)), 3, (0, 0, 255), -1)
        return frame

    def feed_audio(self, audio_chunk, sample_rate=16000):
        """喂入音频"""
        with self.audio_lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            max_samples = sample_rate * 5  # 保留最近5秒
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

            if len(self.audio_buffer) >= sample_rate * 0.5:
                features = extract_mfcc(self.audio_buffer)
                self.latest_audio_feat = prepare_dinet_input(features)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


# === 测试: 视频文件处理 ===

def test_video_file():
    """用视频文件测试完整管线"""
    video_path = "/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/example/video.mp4"
    if not os.path.exists(video_path):
        print(f"视频不存在: {video_path}")
        # 尝试找任意视频
        test_dir = "/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/example"
        if os.path.isdir(test_dir):
            for f in os.listdir(test_dir):
                if f.endswith(('.mp4', '.mov', '.avi', '.webm')):
                    video_path = os.path.join(test_dir, f)
                    break
            else:
                print("未找到测试视频")
                return
        else:
            return

    print(f"测试视频: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  分辨率: {width}x{height}, FPS: {fps_video:.1f}, 帧数: {frame_count}")

    pipeline = StreamingPipeline(detect_interval=5)
    pipeline.start()

    # 提取音频
    import subprocess
    audio_path = "/tmp/pipeline_test_audio.wav"
    subprocess.run(
        f"ffmpeg -y -i '{video_path}' -ac 1 -ar 16000 -t 10 {audio_path}",
        shell=True, capture_output=True
    )
    if os.path.exists(audio_path):
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        pipeline.feed_audio(audio)
        print(f"  音频已加载: {len(audio)/sr:.1f}s")

    # 输出视频
    output_path = "/tmp/pipeline_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

    # 处理循环
    processed = 0
    max_frames = min(frame_count, 150)

    print(f"\n开始处理 {max_frames} 帧...")
    t_start = time.perf_counter()

    while pipeline.running and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)
        out_writer.write(result)
        processed += 1

        if processed % 30 == 0:
            elapsed = time.perf_counter() - t_start
            fps = processed / elapsed
            print(f"\r  帧 {processed}/{max_frames} | 实际 {fps:.1f} FPS | "
                  f"人脸: {pipeline.last_bbox is not None}", end="")

    cap.release()
    out_writer.release()
    pipeline.stop()

    elapsed = time.perf_counter() - t_start
    print(f"\n\n处理完成:")
    print(f"  总帧数: {processed}")
    print(f"  总耗时: {elapsed:.1f}s ({processed/elapsed:.1f} FPS 实际吞吐)")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    test_video_file()
