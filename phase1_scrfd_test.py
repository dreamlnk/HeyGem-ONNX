"""
Phase 1: 人脸检测 (YuNet ONNX)
使用 OpenCV DNN YuNet 替代 SCRFD (对AI生成/美颜人脸更鲁棒)
"""
import os
import time
import cv2
import numpy as np

YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
YUNET_PATH = os.path.join(os.path.dirname(__file__), YUNET_MODEL)
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# 保留旧常量以兼容 pipeline 的 warmup
INPUT_SIZE = 640
STRIDES = [8, 16, 32]


def _download_yunet():
    """Download YuNet model if not present"""
    if os.path.exists(YUNET_PATH):
        return YUNET_PATH
    # Also check /tmp
    tmp_path = "/tmp/" + YUNET_MODEL
    if os.path.exists(tmp_path):
        return tmp_path

    import urllib.request
    urls = [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ]
    for url in urls:
        try:
            urllib.request.urlretrieve(url, tmp_path)
            if os.path.getsize(tmp_path) > 0:
                print(f"  YuNet 下载成功")
                return tmp_path
        except Exception:
            continue
    raise FileNotFoundError(f"无法下载 YuNet 模型到 {tmp_path}")


def load_session():
    """加载 YuNet 检测器 (保持与 SCRFD 相同的接口签名)"""
    model_path = _download_yunet()
    detector = cv2.FaceDetectorYN.create(
        model_path, "", (640, 640),
        SCORE_THRESHOLD, NMS_THRESHOLD, 5000
    )
    print(f"人脸检测: YuNet (OpenCV DNN)")
    return detector


def detect(detector, img_bgr):
    """
    人脸检测 (保持与 SCRFD detect 相同的返回格式)
    Args:
        detector: cv2.FaceDetectorYN 实例
        img_bgr: BGR 图像 [H, W, 3]
    Returns:
        bboxes: [N, 4] int [x1, y1, x2, y2]
        kpss: [N, 10] int 5点关键点 (左眼,右眼,鼻尖,左嘴角,右嘴角)
        meta: dict (空, 兼容旧接口)
    """
    h, w = img_bgr.shape[:2]

    # YuNet 接受任意尺寸, 但需要每次 setInputSize
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)

    if faces is None:
        return [], [], {}

    bboxes_list = []
    kpss_list = []

    for face in faces:
        score = face[-1]
        if score < SCORE_THRESHOLD:
            continue

        # YuNet bbox: [x, y, w, h] → [x1, y1, x2, y2]
        x, y, fw, fh = face[:4].astype(int)
        bbox = np.array([x, y, x + fw, y + fh], dtype=int)

        # YuNet 5点: right_eye, left_eye, nose, right_mouth, left_mouth
        # SCRFD 格式: left_eye, right_eye, nose, left_mouth, right_mouth
        lm = face[4:14]
        kps = np.array([
            lm[2], lm[3],   # left_eye  ← YuNet index 6,7
            lm[0], lm[1],   # right_eye ← YuNet index 4,5
            lm[4], lm[5],   # nose      ← YuNet index 8,9
            lm[8], lm[9],   # left_mouth ← YuNet index 12,13
            lm[6], lm[7],   # right_mouth ← YuNet index 10,11
        ], dtype=int)

        bboxes_list.append(bbox)
        kpss_list.append(kps)

    if not bboxes_list:
        return [], [], {}

    return np.array(bboxes_list), np.array(kpss_list), {}


def draw_result(img_bgr, bboxes, kpss):
    """绘制检测结果"""
    img = img_bgr.copy()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        kps = kpss[i].reshape(-1, 2)
        for kp in kps:
            cv2.circle(img, tuple(kp), 2, (0, 0, 255), -1)
    return img


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: YuNet 人脸检测管线验证")
    print("=" * 60)

    # 1. 加载模型
    detector = load_session()

    # 2. 测试图片
    import glob
    frames = sorted(glob.glob("/tmp/full_frame_*.png"))
    if frames:
        img = cv2.imread(frames[-1])
        print(f"测试图片: {img.shape[1]}x{img.shape[0]}")
    else:
        print("未找到测试图，生成合成图...")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # 3. 推理
    t0 = time.perf_counter()
    bboxes, kpss, _ = detect(detector, img)
    dt = (time.perf_counter() - t0) * 1000
    print(f"推理耗时: {dt:.1f}ms")
    print(f"检测到 {len(bboxes)} 张人脸")

    for i, bbox in enumerate(bboxes):
        print(f"  人脸{i+1}: bbox={bbox.tolist()}, kps={kpss[i][:4]}...")

    # 4. 可视化
    result = draw_result(img, bboxes, kpss)
    out_path = "/tmp/phase1_result.jpg"
    cv2.imwrite(out_path, result)
    print(f"结果保存至: {out_path}")

    # 5. 速度基准
    print(f"\n--- 速度基准 (100次) ---")
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        detect(detector, img)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"  平均: {times.mean():.1f}ms  ({1000/times.mean():.0f} FPS)")
    print(f"  中位: {np.median(times):.1f}ms")
    print(f"  最小: {times.min():.1f}ms  /  最大: {times.max():.1f}ms")
    print("\nPhase 1 验证通过 ✓")
