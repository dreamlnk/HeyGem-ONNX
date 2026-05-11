"""
Phase 2a: 人脸解析 (BiseNet) ONNX 模型验证
用于分割人脸区域（皮肤、眉毛、眼睛、鼻子、嘴巴等）
"""
import os
import time
import cv2
import numpy as np
import onnxruntime

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "pretrain_models/face_lib/face_parsing/79999_iter.onnx"
)


def load_session():
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(
        MODEL_PATH, opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"设备: {session.get_providers()}")
    return session


def inspect_model(session):
    print("\n--- 输入 ---")
    for inp in session.get_inputs():
        print(f"  name={inp.name}, shape={inp.shape}, type={inp.type}")
    print("\n--- 输出 ---")
    for out in session.get_outputs():
        print(f"  name={out.name}, shape={out.shape}, type={out.type}")


def preprocess(img_bgr, target_size=512):
    """人脸解析预处理: resize到512x512, BGR->RGB, normalize"""
    img = cv2.resize(img_bgr, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = img.astype(np.float32) / 255.0
    blob = (blob - 0.5) / 0.5  # normalize to [-1, 1]
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return blob, img


def parse_face(session, img_bgr):
    """人脸解析推理"""
    blob, vis_img = preprocess(img_bgr)

    t0 = time.perf_counter()
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    dt = (time.perf_counter() - t0) * 1000

    # 输出是 [1, num_classes, 512, 512]
    parsing = outputs[0].squeeze(0)  # [num_classes, 512, 512]
    parsing = parsing.argmax(axis=0)  # [512, 512] 每个像素的类别

    return parsing, vis_img, dt


def visualize_parsing(vis_img, parsing):
    """可视化分割结果"""
    # BiseNet 19类颜色映射 (CelebAMask-HQ)
    colors = np.array([
        [0, 0, 0],        # 0: background
        [204, 0, 0],      # 1: skin
        [76, 153, 0],     # 2: nose
        [204, 204, 0],    # 3: eye_g
        [51, 51, 255],    # 4: l_eye
        [204, 0, 204],    # 5: r_eye
        [0, 255, 255],    # 6: l_brow
        [255, 0, 255],    # 7: r_brow
        [0, 255, 0],      # 8: l_ear
        [255, 153, 51],   # 9: r_ear
        [153, 76, 0],     # 10: mouth (interior)
        [255, 204, 153],  # 11: u_lip
        [51, 0, 0],       # 12: l_lip
        [0, 102, 204],    # 13: hair
        [102, 0, 102],    # 14: hat
        [255, 255, 153],  # 15: ear_r
        [0, 0, 204],      # 16: neck_l
        [204, 153, 255],  # 17: neck
        [102, 102, 51],   # 18: cloth
    ], dtype=np.uint8)

    h, w = vis_img.shape[:2]
    parsing_resized = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = vis_img.copy()
    for i in range(1, 19):
        mask = parsing_resized == i
        overlay[mask] = (overlay[mask] * 0.4 + colors[i] * 0.6).astype(np.uint8)

    # 重点高亮嘴巴区域 (类别 10,11,12)
    mouth_mask = np.isin(parsing_resized, [10, 11, 12])
    overlay[mouth_mask] = overlay[mouth_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5

    return overlay


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2a: 人脸解析 (BiseNet) ONNX 模型验证")
    print("=" * 60)

    session = load_session()
    inspect_model(session)

    # 从视频提取人脸区域测试
    test_img = "/tmp/test_frame.jpg"
    img = cv2.imread(test_img)
    if img is None:
        print("未找到测试图，生成合成图...")
        img = np.ones((512, 512, 3), dtype=np.uint8) * 128

    # 先用SCRFD检测人脸，取最大人脸区域
    from phase1_scrfd_test import detect as scrfd_detect, load_session as scrfd_load
    scrfd = scrfd_load()
    bboxes, kpss, _ = scrfd_detect(scrfd, img)

    if len(bboxes) > 0:
        # 取最大人脸，扩大裁切区域
        x1, y1, x2, y2 = bboxes[0]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        size = max(x2 - x1, y2 - y1) * 1.5
        h, w = img.shape[:2]
        x1 = max(0, int(cx - size // 2))
        x2 = min(w, int(cx + size // 2))
        y1 = max(0, int(cy - size // 2))
        y2 = min(h, int(cy + size // 2))
        face_img = img[y1:y2, x1:x2]
        print(f"人脸区域: {face_img.shape[1]}x{face_img.shape[0]}")
    else:
        face_img = cv2.resize(img, (512, 512))

    parsing, vis_img, dt = parse_face(session, face_img)
    print(f"推理耗时: {dt:.1f}ms")
    print(f"分割类别数: {parsing.max() + 1}")

    # 统计各类别占比
    unique, counts = np.unique(parsing, return_counts=True)
    for cls, cnt in zip(unique[:8], counts[:8]):
        names = {0:"背景", 1:"皮肤", 2:"鼻子", 3:"眼睛", 4:"左眼", 5:"右眼", 6:"左眉", 7:"右眉"}
        print(f"  {names.get(cls, cls)}: {cnt/parsing.size*100:.1f}%")

    overlay = visualize_parsing(vis_img, parsing)
    out_path = "/tmp/phase2_parsing.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"结果保存至: {out_path}")

    # 速度基准
    print(f"\n--- 速度基准 (50次) ---")
    times = []
    s = face_img.shape
    for _ in range(50):
        blob = np.random.randn(1, 3, 512, 512).astype(np.float32)
        t0 = time.perf_counter()
        session.run(None, {session.get_inputs()[0].name: blob})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"  平均: {times.mean():.1f}ms  ({1000/times.mean():.0f} FPS)")
    print(f"  中位: {np.median(times):.1f}ms")
    print("\nPhase 2a 完成 ✓")
