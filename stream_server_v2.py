"""
流式管线服务端 v2 (WSL侧)
接收逐帧JPEG + 音频块，运行完整管线，返回渲染帧
"""
import os, sys, io, time, json, threading
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, Response

sys.path.insert(0, os.path.dirname(__file__))

from pipeline_complete import StreamingPipeline, align_face, preprocess_face_for_dinet, inverse_affine_transform, REFERENCE_POINTS
from phase1_scrfd_test import load_session as scrfd_load, detect as scrfd_detect
from phase2_audio_feature import extract_mfcc, prepare_dinet_input

app = Flask(__name__)

# 全局管线实例 (延迟初始化)
pipeline = None
pipeline_lock = threading.Lock()
stats = {"frames_processed": 0, "total_time": 0, "errors": 0}


def get_pipeline():
    global pipeline
    if pipeline is None:
        with pipeline_lock:
            if pipeline is None:
                print("初始化管线...")
                pipeline = StreamingPipeline(detect_interval=5)
                pipeline.start()
                print("管线就绪")
    return pipeline


@app.route("/health")
def health():
    p = pipeline
    status = "ready" if p and p.source_face_tensor is not None else "warming"
    return jsonify({
        "status": status,
        "stats": stats,
        "pipeline_ready": p is not None,
        "source_face_set": p is not None and p.source_face_tensor is not None,
    })


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """接收JPEG帧 + WAV音频块，返回渲染后JPEG帧"""
    t0 = time.perf_counter()

    if "frame" not in request.files:
        return jsonify({"error": "missing frame"}), 400

    try:
        p = get_pipeline()

        # 解码JPEG帧 + 缩放到处理分辨率
        frame_file = request.files["frame"]
        frame_bytes = frame_file.read()
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid frame"}), 400
        # 统一缩放到720p (如果客户端发了更大分辨率)
        h, w = frame.shape[:2]
        if w != 1280 or h != 720:
            frame = cv2.resize(frame, (1280, 720))

        # 处理音频 (如果有)
        if "audio" in request.files:
            audio_bytes = request.files["audio"].read()
            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            if len(audio_np) > 0:
                p.feed_audio(audio_np)

        # 确保音频特征已初始化 (无音频时使用零特征)
        if p.latest_audio_feat is None:
            p.latest_audio_feat = np.zeros((256, 256), dtype=np.float32)

        # 运行管线
        result = p.process_frame(frame)

        # 编码为JPEG返回
        _, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        dt = (time.perf_counter() - t0) * 1000

        stats["frames_processed"] += 1
        stats["total_time"] += dt

        return Response(jpeg.tobytes(), mimetype="image/jpeg")

    except Exception as e:
        stats["errors"] += 1
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """重置管线状态 (切换场景/人脸时调用)"""
    global pipeline
    with pipeline_lock:
        if pipeline:
            pipeline.stop()
        pipeline = None
    return jsonify({"status": "reset"})


@app.route("/stats")
def get_stats():
    s = dict(stats)
    if stats["frames_processed"] > 0:
        s["avg_ms"] = round(stats["total_time"] / stats["frames_processed"], 1)
        s["avg_fps"] = round(1000 / s["avg_ms"], 1)
    p = pipeline
    if p:
        s["has_face"] = p.last_bbox is not None
        s["source_set"] = p.source_face_tensor is not None
        s["audio_ready"] = p.latest_audio_feat is not None
        s["frame_idx"] = p.frame_idx
    return jsonify(s)


if __name__ == "__main__":
    print("=" * 50)
    print("流式管线服务端 v2")
    print("端口: 7861")
    print("=" * 50)

    # 预加载管线 (避免首次请求超时)
    print("预加载管线...")
    get_pipeline()
    print("管线就绪，开始监听...")

    app.run(host="0.0.0.0", port=7861, debug=False, threaded=True)
