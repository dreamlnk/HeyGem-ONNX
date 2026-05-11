import os
import sys
import uuid
import cv2
from flask import Flask, request, render_template_string, send_file, jsonify
import service.trans_dh_service
from y_utils.config import GlobalConfig
from y_utils.logger import logger

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>HeyGem ONNX - 数字人对口型</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
        .container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
        h1 { font-size: 28px; margin-bottom: 8px; }
        .subtitle { color: #94a3b8; margin-bottom: 32px; }
        .card { background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
        .card h2 { font-size: 18px; margin-bottom: 16px; color: #38bdf8; }
        input[type="file"] { display: block; width: 100%; padding: 12px; background: #334155; border: 2px dashed #475569; border-radius: 8px; color: #e2e8f0; cursor: pointer; margin-bottom: 12px; }
        input[type="file"]:hover { border-color: #38bdf8; }
        button { background: #2563eb; color: white; border: none; padding: 12px 32px; border-radius: 8px; font-size: 16px; cursor: pointer; width: 100%; }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #475569; cursor: not-allowed; }
        #status { margin-top: 16px; padding: 12px; border-radius: 8px; display: none; }
        #status.info { display: block; background: #1e3a5f; color: #93c5fd; }
        #status.success { display: block; background: #14532d; color: #86efac; }
        #status.error { display: block; background: #7f1d1d; color: #fca5a5; }
        #result { margin-top: 20px; display: none; }
        #result video { width: 100%; border-radius: 8px; }
        .file-name { font-size: 13px; color: #94a3b8; margin-top: 4px; }
        .tip { color: #94a3b8; font-size: 13px; margin-top: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>HeyGem ONNX 数字人</h1>
        <p class="subtitle">音频驱动 · 口型同步 · GPU 加速</p>

        <div class="card">
            <h2>1. 上传音频文件</h2>
            <input type="file" id="audioInput" accept="audio/*">
            <div class="file-name" id="audioName"></div>
            <p class="tip">支持 WAV, MP3 等格式</p>
        </div>

        <div class="card">
            <h2>2. 上传视频文件</h2>
            <input type="file" id="videoInput" accept="video/*">
            <div class="file-name" id="videoName"></div>
            <p class="tip">包含人脸的视频，支持 MP4, MOV 等格式</p>
        </div>

        <button id="submitBtn" onclick="submitTask()">开始生成</button>

        <div id="status"></div>
        <div id="result">
            <h2 style="margin-bottom:12px">生成结果</h2>
            <video id="resultVideo" controls autoplay loop></video>
        </div>
    </div>

    <script>
        document.getElementById('audioInput').onchange = function(e) {
            document.getElementById('audioName').textContent = e.target.files[0]?.name || '';
        };
        document.getElementById('videoInput').onchange = function(e) {
            document.getElementById('videoName').textContent = e.target.files[0]?.name || '';
        };

        async function submitTask() {
            const audioFile = document.getElementById('audioInput').files[0];
            const videoFile = document.getElementById('videoInput').files[0];
            const status = document.getElementById('status');
            const result = document.getElementById('result');
            const btn = document.getElementById('submitBtn');

            if (!audioFile || !videoFile) {
                status.className = 'error';
                status.textContent = '请先上传音频和视频文件';
                return;
            }

            const formData = new FormData();
            formData.append('audio', audioFile);
            formData.append('video', videoFile);

            btn.disabled = true;
            btn.textContent = '生成中，请等待...';
            status.className = 'info';
            status.textContent = '正在处理，大约需要 1-3 分钟...';
            result.style.display = 'none';

            try {
                const resp = await fetch('/generate', { method: 'POST', body: formData });
                const data = await resp.json();

                if (data.success) {
                    status.className = 'success';
                    status.textContent = '生成完成！耗时: ' + data.cost + ' 秒';
                    result.style.display = 'block';
                    document.getElementById('resultVideo').src = data.video_url + '?t=' + Date.now();
                } else {
                    status.className = 'error';
                    status.textContent = '错误: ' + data.error;
                }
            } catch (err) {
                status.className = 'error';
                status.textContent = '请求失败: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = '开始生成';
            }
        }
    </script>
</body>
</html>
"""

task_instance = None


def get_task():
    global task_instance
    if task_instance is None:
        task_instance = service.trans_dh_service.TransDhTask()
    return task_instance


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    import time
    start = time.time()

    audio_file = request.files.get('audio')
    video_file = request.files.get('video')

    if not audio_file or not video_file:
        return jsonify({'success': False, 'error': '缺少音频或视频文件'})

    work_id = str(uuid.uuid1())
    upload_dir = os.path.join(GlobalConfig.instance().temp_dir, work_id)
    os.makedirs(upload_dir, exist_ok=True)

    audio_path = os.path.join(upload_dir, 'input_audio.wav')
    video_path = os.path.join(upload_dir, 'input_video.mp4')
    audio_file.save(audio_path)
    video_file.save(video_path)

    try:
        task = get_task()
        code = work_id
        task.task_dic[code] = ""
        task.work(audio_path, video_path, code, 0, 0, 0, 0)

        result_path = task.task_dic[code][2]
        cost = round(time.time() - start, 1)

        return jsonify({
            'success': True,
            'video_url': '/result/' + os.path.basename(result_path),
            'cost': cost
        })
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/result/<filename>')
def result(filename):
    result_dir = GlobalConfig.instance().result_dir
    return send_file(os.path.join(result_dir, filename))


if __name__ == '__main__':
    logger.info("启动 HeyGem ONNX Web 服务...")
    app.run(host='0.0.0.0', port=7860, debug=False)
