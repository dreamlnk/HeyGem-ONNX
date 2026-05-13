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
from phase2_audio_feature import extract_mfcc, prepare_dinet_input, prepare_logmel_dinet_input, sr_samples

CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"

# === 256x256 标准人脸参考点 (Dlib对齐, 匹配DINet训练数据) ===
# 训练数据: {img_size}_dlib_crop + eye mask row 20-70 + mouth mask row ~83-246
# Dlib标准: 左眼(0.35,0.35)*256, 右眼(0.65,0.35)*256
REFERENCE_POINTS = np.array([
    [89.60, 89.60],    # left eye
    [166.40, 89.60],   # right eye
    [128.00, 132.00],  # nose
    [89.60, 185.60],   # left mouth corner
    [166.40, 185.60],  # right mouth corner
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


def inverse_affine_transform(img_256, original_frame, M):
    """将DINet输出(256x256)逆变换贴回原始帧 — 全脸合成 + 内容感知遮罩"""
    h, w = original_frame.shape[:2]

    M_inv = cv2.invertAffineTransform(M)
    corners_256 = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype=np.float32)
    corners_orig = cv2.transform(corners_256.reshape(1, -1, 2), M_inv).reshape(-1, 2)

    x_min = max(0, int(corners_orig[:, 0].min()))
    x_max = min(w, int(corners_orig[:, 0].max()) + 1)
    y_min = max(0, int(corners_orig[:, 1].min()))
    y_max = min(h, int(corners_orig[:, 1].max()) + 1)
    roi_w, roi_h = x_max - x_min, y_max - y_min

    if roi_w <= 0 or roi_h <= 0:
        return original_frame

    M_roi = M_inv.copy()
    M_roi[0, 2] -= x_min
    M_roi[1, 2] -= y_min

    # Warp渲染人脸
    warped_face = cv2.warpAffine(img_256, M_roi, (roi_w, roi_h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 内容感知遮罩: 模型输出白色区域(>200)不参与合成, 仅保留有内容的区域
    gray_256 = cv2.cvtColor(img_256, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 使用更宽松的阈值(200而非235)，让更多模型输出参与合成
    content_mask = 1.0 - np.clip(gray_256 / 255.0, 0, 1)
    content_mask = np.clip(content_mask * 3.0, 0, 1)  # 增强非白色区域
    content_mask = cv2.GaussianBlur(content_mask, (31, 31), 10)
    warped_content = cv2.warpAffine(content_mask, M_roi, (roi_w, roi_h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    warped_content = np.nan_to_num(warped_content, nan=0.0).clip(0, 1)

    # Alpha混合: 直接使用内容遮罩作为alpha
    final_alpha = warped_content[..., np.newaxis]
    final_alpha = np.clip(final_alpha, 0, 1)

    result = original_frame.copy()
    roi = result[y_min:y_max, x_min:x_max].astype(np.float32)
    warped_face_f = np.nan_to_num(warped_face.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)

    blended = warped_face_f * final_alpha + roi * (1.0 - final_alpha)
    blended = np.nan_to_num(blended, nan=0.0, posinf=255.0, neginf=0.0)
    result[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


def create_eye_mask(size=256):
    """创建额头遮罩 — 匹配DINet训练数据(dlib对齐, mask row 20-70)"""
    mask = np.ones((size, size), dtype=np.uint8) * 255
    mask[20:70, 55:-55] = 0
    return mask


def create_mouth_mask(size=256):
    """创建下半脸遮罩 mask_B — 匹配DINet训练(dlib对齐, start=128-45, end~246)"""
    mask = np.ones((size, size), dtype=np.uint8) * 255
    mask[83:246, 30:-30] = 0
    return mask


def preprocess_face_for_dinet(aligned_bgr, apply_eye_mask=True):
    """
    对齐人脸 → DINet输入格式 (保持BGR, DINet训练时使用OpenCV BGR)
    """
    img = aligned_bgr.copy()
    if apply_eye_mask:
        eye_mask = create_eye_mask()
        img = cv2.bitwise_and(img, img, mask=eye_mask)
    # 归一化到 [-1, 1]: (x/255 - 0.5) / 0.5
    tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


def preprocess_mask_B_for_dinet(aligned_bgr):
    """
    创建mask_B: 对齐人脸 + 嘴部遮罩 (保持BGR)
    """
    img = aligned_bgr.copy()
    mouth_mask = create_mouth_mask()
    img = cv2.bitwise_and(img, img, mask=mouth_mask)
    tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


def denormalize_output(tensor):
    """
    DINet输出 [0, 1] → uint8 BGR (模型输出BGR, 无需转换)
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.float().cpu().numpy()
    img = np.clip(img, 0, 1).transpose(1, 2, 0)  # CHW → HWC
    img = (img * 255).astype(np.uint8)
    return img


class DINetInferenceEngine:
    """DINet推理引擎，管理模型生命周期和推理"""

    def __init__(self):
        from landmark2face_wy.models.networks import DINetV1

        print("  加载 DINetV1...")
        self.model = DINetV1(source_channel=3, ref_channel=3, audio_channel=256)
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        model_state = self.model.state_dict()
        ckpt_state = ckpt["face_G"]
        missing = set(model_state.keys()) - set(ckpt_state.keys())
        extra = set(ckpt_state.keys()) - set(model_state.keys())
        matching = set(model_state.keys()) & set(ckpt_state.keys())
        if missing:
            print(f"  [WARN] 缺失权重: {len(missing)} keys (strict=False会跳过)")
            for k in sorted(missing)[:5]:
                print(f"    - {k}")
        if extra:
            print(f"  [INFO] 多余权重: {len(extra)} keys")
        print(f"  [INFO] 匹配权重: {len(matching)}/{len(model_state)} keys")
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

    def __init__(self, detect_interval=3, test_audio=False):
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

        # 音频缓冲 + 对数梅尔谱特征提取
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_lock = threading.Lock()
        self.dinet_lock = threading.Lock()  # DINet CUDA推理锁 (防止多客户端并发覆盖共享tensor)
        self.latest_audio_feat = None
        self._last_rendered = None
        self.test_audio = test_audio

        # 预加载对数梅尔谱模块
        print("  加载对数梅尔谱提取...")
        try:
            from phase2_audio_feature import _get_wm
            _get_wm()
            print("  对数梅尔谱提取就绪 ✓")
        except Exception as e:
            print(f"  [警告] 加载失败: {e}")

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
        # YuNet warmup
        dummy = np.ones((640, 640, 3), dtype=np.uint8) * 128
        for _ in range(3):
            self.scrfd.setInputSize((640, 640))
            self.scrfd.detect(dummy)
        # DINet warmup
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

        # 1. 人脸检测 (跳帧)
        do_detect = (self.frame_idx % self.detect_interval == 0) or (self.last_bbox is None)
        if do_detect:
            bboxes, kpss, _ = scrfd_detect(self.scrfd, frame_bgr)

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

        if self.last_bbox is None:
            return frame_bgr

        if not self._valid_detection(self.last_bbox, self.last_kps, frame_bgr):
            self.last_bbox = None
            self.last_kps = None
            return frame_bgr

        # 2. 人脸对齐
        if self.last_kps is None:
            return frame_bgr

        aligned_face, M = align_face(frame_bgr, self.last_kps)
        if aligned_face is None:
            return frame_bgr
        self.last_M = M

        # 3. 设置身份参考帧 (第一帧)
        if self.source_face_tensor is None:
            self.set_source_face(frame_bgr, self.last_bbox, self.last_kps)

        # 3.5. test_audio模式: 每帧更新交替测试特征
        if self.test_audio:
            self.feed_test_audio()

        # 4. 准备DINet输入
        if self.source_face_tensor is None or self.latest_audio_feat is None:
            return frame_bgr

        ref_tensor = preprocess_mask_B_for_dinet(aligned_face)

        # 5. DINet推理
        with self.dinet_lock:
            rendered_256 = self.dinet.infer(
                self.source_face_tensor,
                ref_tensor,
                self.latest_audio_feat,
            )

        # DEBUG: 跟踪渲染输出变化 + 定期保存快照
        if self._last_rendered is not None:
            diff = np.abs(rendered_256.astype(np.float32) - self._last_rendered.astype(np.float32)).mean()
            if self.frame_idx % 15 == 0:
                print(f"\r[DINet] 帧{self.frame_idx} 帧间差异={diff:.2f}  ", end="", flush=True,
                      file=__import__('sys').stderr)
        self._last_rendered = rendered_256.copy()
        if self.frame_idx % 15 == 1:
            cv2.imwrite("/tmp/dinet_rendered_256.png", rendered_256)
            cv2.imwrite("/tmp/dinet_aligned_ref.png", aligned_face)
            if self.frame_idx == 1:
                cv2.imwrite("/tmp/dinet_aligned_src.png",
                            ((self.source_face_tensor.cpu().numpy() * 0.5 + 0.5) * 255)
                            .transpose(1, 2, 0).astype(np.uint8))

        # 6. 合成回原始帧
        result = inverse_affine_transform(rendered_256, frame_bgr, self.last_M)

        # DEBUG: 每75帧保存合成结果
        if self.frame_idx % 75 == 1:
            cv2.imwrite("/tmp/dinet_composited.png", result)
            # Save alpha visualization
            gray_r = cv2.cvtColor(rendered_256, cv2.COLOR_BGR2GRAY)
            content_viz = (np.clip(gray_r, 0, 255)).astype(np.uint8)
            cv2.imwrite("/tmp/dinet_content.png", content_viz)

        return result

    def _valid_detection(self, bbox, kps, frame):
        """过滤虚假检测: bbox必须至少部分在画面内且尺寸合理"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        # bbox必须至少部分在画面内 (x2>0, x1<w)
        if x2 < 1 or x1 > w - 1 or y2 < 1 or y1 > h - 1:
            import sys as _sys
            print(f"[VALIDATE] REJECT boundary: bbox={bbox} frame={w}x{h}", flush=True, file=_sys.stderr)
            return False
        # 尺寸合理: 20~120%画面尺寸
        if bw < 20 or bh < 20 or bw > w * 1.2 or bh > h * 1.2:
            import sys as _sys
            print(f"[VALIDATE] REJECT size: bbox={bbox} bw={bw} bh={bh} frame={w}x{h} limit_w={w*1.2:.0f} limit_h={h*1.2:.0f}", flush=True, file=_sys.stderr)
            return False
        # bbox不能是退化的 (x1 >= x2 or y1 >= y2)
        if bw <= 0 or bh <= 0:
            return False
        if kps is not None:
            if kps.ndim == 1:
                kps = kps.reshape(-1, 2)
            for kp in kps:
                if not (-w < kp[0] < 2*w and -h < kp[1] < 2*h):
                    import sys as _sys
                    print(f"[VALIDATE] REJECT kps: kp={kp} frame={w}x{h}", flush=True, file=_sys.stderr)
                    return False
        return True

    def _has_skin(self, frame_bgr, bbox):
        """验证检测框内是否包含足够比例的肤色像素"""
        import sys as _sys
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return False
        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, (0, 15, 50), (25, 180, 255))
        ratio = skin.sum() / 255 / (roi.shape[0] * roi.shape[1])
        result = ratio > 0.05
        if not result:
            print(f"[SKIN] REJECT: skin_ratio={ratio:.3f} bbox={bbox}", flush=True, file=_sys.stderr)
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
        """喂入音频 → 对数梅尔谱 → DINet [256,256]"""
        if self.test_audio:
            return  # test_audio模式下忽略客户端音频
        with self.audio_lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            max_samples = sample_rate * 10  # 保留最近10秒
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

            if len(self.audio_buffer) >= sample_rate * 0.5:
                self.latest_audio_feat = prepare_logmel_dinet_input(self.audio_buffer)
                if self.frame_idx % 15 == 0:
                    f = self.latest_audio_feat
                    roi = f[:, :80]
                    nz = roi[roi != 0]
                    print(f"\r[音频特征] shape={f.shape} 非零区域: mean={nz.mean():.1f} std={nz.std():.1f} "
                          f"min={nz.min():.1f} max={nz.max():.1f} count={len(nz)}",
                          flush=True, file=__import__('sys').stderr)

    def feed_test_audio(self):
        """测试模式: 预计算交替音频特征，每15帧切换(张嘴/闭嘴)"""
        if not hasattr(self, '_test_feats'):
            import math
            from phase2_audio_feature import prepare_logmel_dinet_input

            sr = 16000
            dur = 3.0
            samples = int(sr * dur)
            tt = np.linspace(0, dur, samples, endpoint=False, dtype=np.float32)

            # 特征A: 高能量扫频 (模拟说话)
            freq = 100.0 + 700.0 * (tt / dur)
            sweep_a = np.sin(2.0 * math.pi * freq * tt).astype(np.float32) * 0.8

            # 特征B: 静音
            sweep_b = np.zeros(samples, dtype=np.float32)

            self._test_feats = [
                prepare_logmel_dinet_input(sweep_a),
                prepare_logmel_dinet_input(sweep_b),
            ]
            print(f"  测试音频特征预计算完成", flush=True, file=__import__('sys').stderr)

        idx = (self.frame_idx // 15) % 2
        self.latest_audio_feat = self._test_feats[idx]

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
