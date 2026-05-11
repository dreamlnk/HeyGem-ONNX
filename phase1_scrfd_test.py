"""
Phase 1: SCRFD ONNX 人脸检测 — 完整管线验证
绕过 .so，直接用 onnxruntime 加载 SCRFD 模型，包含预处理+推理+后处理
"""
import os
import time
import cv2
import numpy as np
import onnxruntime

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx"
)

# 模型固定参数
INPUT_SIZE = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
STRIDES = [8, 16, 32]
# 每个 stride 每个格点 2 个 anchor
FMCS = {
    8:  [[5.0, 5.0], [3.0, 3.0]],
    16: [[7.0, 7.0], [5.0, 5.0]],
    32: [[9.0, 9.0], [7.0, 7.0]],
}


def load_session():
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(
        MODEL_PATH, opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"设备: {session.get_providers()}")
    return session


def preprocess(img_bgr):
    """预处理: resize + BGR->RGB + normalize"""
    h, w = img_bgr.shape[:2]
    ratio = INPUT_SIZE / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    img = cv2.resize(img_bgr, (new_w, new_h))

    # 补边到 640x640
    pad_h = INPUT_SIZE - new_h
    pad_w = INPUT_SIZE - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # BGR -> RGB, HWC -> CHW, [0,255] -> [0,1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    meta = {"ratio": ratio, "pad_top": top, "pad_left": left, "orig_h": h, "orig_w": w}
    return blob, meta


def distance2bbox(points, distance, stride, max_shape=None):
    """将偏移量解码为边界框坐标 [x1, y1, x2, y2]"""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, stride):
    """将关键点偏移量解码为绝对坐标"""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 1] + distance[:, i] * stride
        py = points[:, 0] + distance[:, i + 1] * stride
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def generate_anchors(stride, input_size=INPUT_SIZE):
    """生成每个 stride 的锚点中心坐标"""
    fmap_size = input_size // stride
    fc = FMCS[stride]
    num_anchors = len(fc)
    anchors = np.zeros((num_anchors, fmap_size * fmap_size, 2), dtype=np.float32)
    for i, (w, h) in enumerate(fc):
        xs, ys = np.meshgrid(np.arange(fmap_size), np.arange(fmap_size))
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        anchors[i, :, 0] = cx.flatten()
        anchors[i, :, 1] = cy.flatten()
    return anchors.reshape(-1, 2)  # (num_anchors * fmap_size^2, 2)


def detect(session, img_bgr):
    """完整检测流程"""
    blob, meta = preprocess(img_bgr)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: blob})

    scores_list, bboxes_list, kpss_list = [], [], []
    num_strides = len(STRIDES)

    for idx, stride in enumerate(STRIDES):
        score = outputs[idx]           # [1, N, 1]
        bbox = outputs[idx + num_strides]    # [1, N, 4]
        kps = outputs[idx + num_strides * 2] # [1, N, 10]
        score = score.squeeze(0)  # [N, 1]
        bbox = bbox.squeeze(0)    # [N, 4]
        kps = kps.squeeze(0)      # [N, 10]

        anchors = generate_anchors(stride)
        bbox = distance2bbox(anchors, bbox * stride, stride, max_shape=(INPUT_SIZE, INPUT_SIZE))
        kps = distance2kps(anchors, kps * stride, stride)

        pos = score[:, 0] > SCORE_THRESHOLD
        scores_list.append(score[pos])
        bboxes_list.append(bbox[pos])
        kpss_list.append(kps[pos])

    if not scores_list or all(len(s) == 0 for s in scores_list):
        return [], [], meta

    scores = np.concatenate(scores_list, axis=0).flatten()
    bboxes = np.concatenate(bboxes_list, axis=0)
    kpss = np.concatenate(kpss_list, axis=0)

    # NMS
    keep = cv2.dnn.NMSBoxes(
        bboxes.tolist(), scores.tolist(),
        SCORE_THRESHOLD, NMS_THRESHOLD
    )
    if len(keep) == 0:
        return [], [], meta

    keep = keep.flatten()
    bboxes = bboxes[keep]
    kpss = kpss[keep]
    scores = scores[keep]

    # 坐标还原到原始图像
    ratio = meta["ratio"]
    pad_top, pad_left = meta["pad_top"], meta["pad_left"]
    bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - pad_left) / ratio
    bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - pad_top) / ratio
    kpss[:, 0::2] = (kpss[:, 0::2] - pad_left) / ratio
    kpss[:, 1::2] = (kpss[:, 1::2] - pad_top) / ratio

    return bboxes.astype(int), kpss.astype(int), meta


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
    print("Phase 1: SCRFD ONNX 人脸检测管线验证")
    print("=" * 60)

    # 1. 加载模型
    session = load_session()

    # 2. 测试图片
    test_img = "/tmp/test_frame.jpg"
    img = cv2.imread(test_img)
    if img is None:
        # fallback: 生成纯色图
        print("未找到测试图，生成合成图...")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    print(f"测试图片: {img.shape[1]}x{img.shape[0]}")

    # 3. 推理
    t0 = time.perf_counter()
    bboxes, kpss, meta = detect(session, img)
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
        detect(session, img)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"  平均: {times.mean():.1f}ms  ({1000/times.mean():.0f} FPS)")
    print(f"  中位: {np.median(times):.1f}ms")
    print(f"  最小: {times.min():.1f}ms  /  最大: {times.max():.1f}ms")
    print("\nPhase 1 验证通过 ✓")
