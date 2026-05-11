"""
流式处理服务端 (WSL侧)
接收音频+视频数据，调用管线处理，返回结果帧
使用 HTTP API，方便 Windows 客户端调用
"""
import os
import sys
import uuid
import time
import json
import shutil
import threading
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file

sys.path.insert(0, os.path.dirname(__file__))

import service.trans_dh_service
from y_utils.config import GlobalConfig
from y_utils.logger import logger
from phase1_scrfd_test import load_session as scrfd_load, detect as scrfd_detect

app = Flask(__name__)

task_instance = None


def get_task():
    global task_instance
    if task_instance is None:
        task_instance = service.trans_dh_service.TransDhTask()
    return task_instance


# Monkey-patch write_video 用于帧捕获
captured_frames = {}
frames_lock = threading.Lock()


def write_video_capture(
    output_imgs_queue, temp_dir, result_dir, work_id,
    audio_path, result_queue, width, height, fps,
    watermark_switch=0, digital_auth=0,
):
    """拦截版 write_video: 捕获每一帧并保存"""
    output_mp4 = os.path.join(temp_dir, f"{work_id}-t.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result_path = os.path.join(result_dir, f"{work_id}-r.mp4")
    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    frames = []
    try:
        while True:
            state, reason, value_ = output_imgs_queue.get()
            if type(state) == bool and state is True:
                video_write.release()
                break
            elif type(state) == bool and state is False:
                video_write.release()
                raise RuntimeError(str(reason))
            for result_img in value_:
                video_write.write(result_img)
                frames.append(result_img.copy())

        video_write.release()

        import subprocess
        command = "ffmpeg -loglevel warning -y -i {} -i {} -c:a aac -c:v libx264 -crf 15 -strict -2 {}".format(
            audio_path, output_mp4, result_path
        )
        subprocess.call(command, shell=True)

        with frames_lock:
            captured_frames[work_id] = {
                "frames": frames,
                "result_path": result_path,
                "width": width,
                "height": height,
                "fps": fps,
            }

        result_queue.put([True, result_path])
    except Exception as e:
        logger.error(f"VideoWriter [{work_id}] error: {e}")
        with frames_lock:
            captured_frames[work_id] = {"error": str(e)}
        result_queue.put([False, str(e)])


service.trans_dh_service.write_video = write_video_capture


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    """处理音频+视频文件，返回结果视频"""
    start = time.time()

    audio_file = request.files.get("audio")
    video_file = request.files.get("video")
    if not audio_file or not video_file:
        return jsonify({"success": False, "error": "missing audio or video"})

    work_id = str(uuid.uuid1())
    upload_dir = os.path.join(GlobalConfig.instance().temp_dir, work_id)
    os.makedirs(upload_dir, exist_ok=True)

    audio_path = os.path.join(upload_dir, "input_audio.wav")
    video_path = os.path.join(upload_dir, "input_video.mp4")
    audio_file.save(audio_path)
    video_file.save(video_path)

    try:
        task = get_task()
        code = work_id
        task.task_dic[code] = ""

        # 在线程中运行，设置超时
        result_holder = {}
        error_holder = {}

        def worker():
            try:
                task.work(audio_path, video_path, code, 0, 0, 0, 0)
                result_holder["result"] = task.task_dic.get(code)
            except Exception as e:
                error_holder["error"] = str(e)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=600)

        if thread.is_alive():
            return jsonify({"success": False, "error": "processing timeout"})

        if "error" in error_holder:
            return jsonify({"success": False, "error": error_holder["error"]})

        task_result = result_holder.get("result")
        if not task_result or len(task_result) < 3 or not task_result[2]:
            return jsonify({"success": False, "error": "no output generated"})

        result_path = task_result[2]
        cost = round(time.time() - start, 1)

        # 检查是否有捕获的帧
        frame_count = 0
        with frames_lock:
            if work_id in captured_frames:
                frame_count = len(captured_frames[work_id].get("frames", []))

        return jsonify({
            "success": True,
            "video_url": "/result/" + os.path.basename(result_path),
            "cost": cost,
            "frames_captured": frame_count,
        })

    except Exception as e:
        logger.error(f"Process error: {e}")
        return jsonify({"success": False, "error": str(e)})
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


@app.route("/result/<filename>")
def result(filename):
    return send_file(os.path.join(GlobalConfig.instance().result_dir, filename))


@app.route("/frames/<work_id>/<int:frame_idx>")
def get_frame(work_id, frame_idx):
    """获取单个捕获帧"""
    with frames_lock:
        if work_id not in captured_frames:
            return jsonify({"error": "not found"}), 404
        data = captured_frames[work_id]
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        frames = data["frames"]
        if frame_idx >= len(frames):
            return jsonify({"error": "frame index out of range"}), 404

        frame = frames[frame_idx]
        _, buf = cv2.imencode(".jpg", frame)
        import flask
        return flask.Response(buf.tobytes(), mimetype="image/jpeg")


if __name__ == "__main__":
    logger.info("启动流式处理服务 (端口 7861)...")
    app.run(host="0.0.0.0", port=7861, debug=False)
