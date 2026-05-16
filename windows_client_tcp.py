"""
Windows 客户端 (TCP版) - 原始BGR传输, 零编解码开销
运行:
  python windows_client_tcp.py                      # 摄像头预览
  python windows_client_tcp.py --virtualcam          # 摄像头+OBS虚拟摄像头
  python windows_client_tcp.py --video example/video.mp4           # 视频文件预览
  python windows_client_tcp.py --video example/video.mp4 --virtualcam  # 视频+OBS
  python windows_client_tcp.py --video example/video.mp4 --audio audio.mp3  # 视频+独立音频
"""
import time, queue, struct, socket, threading, argparse, os
import numpy as np
import cv2

# === 配置 ===
WSL_HOST = "127.0.0.1"
WSL_PORT = 17864
WIDTH, HEIGHT = 1280, 720
FRAME_SIZE = WIDTH * HEIGHT * 3
CAMERA_ID = 0
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 0.5


class TCPStreamingClient:
    def __init__(self, video_path=None, use_virtualcam=False, loop=False,
                 use_mic=False, portrait=False, obs_win=False, audio_path=None,
                 size=96):
        self.running = False
        self.video_path = video_path
        self.audio_path = audio_path
        self.loop = loop
        self.use_virtualcam = use_virtualcam
        self.use_mic = use_mic
        self.obs_win = obs_win
        self.portrait = portrait
        self.size = size
        self.width = WIDTH
        self.height = HEIGHT
        self._detect_status = '---'
        self.latency_history = []
        self.frame_count = 0
        self.sock = None
        self.sock_lock = threading.Lock()

    def _connect(self):
        while self.running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                sock.connect((WSL_HOST, WSL_PORT))
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.sock_lock:
                    self.sock = sock
                return True
            except Exception:
                print(f"\r[等待] WSL服务 {WSL_HOST}:{WSL_PORT} ...", end="")
                time.sleep(2)
        return False

    def _draw_hud(self, frame_bgr):
        """Draw semi-transparent HUD at top of frame with all status info."""
        h, w = frame_bgr.shape[:2]
        lines = []
        # Connection + face status
        if self._detect_status == 'NOCONN':
            lines.append(("NO CONN", (0, 0, 255)))
        elif self._detect_status == 'FAIL':
            lines.append(("NO FACE", (0, 0, 255)))
        elif self._detect_status == 'OK':
            lines.append(("FACE OK", (0, 255, 0)))
        else:
            lines.append((f"STATUS: {self._detect_status}", (0, 200, 255)))
        # Mouth delta
        md = getattr(self, '_mouth_delta', 0)
        mc = (0, 255, 0) if md > 3 else (0, 200, 255) if md > 1 else (0, 0, 255)
        lines.append((f"Mouth d: {md:.1f}/255", mc))
        # FPS / latency
        if self.latency_history:
            avg = np.mean(self.latency_history)
            fps = 1000 / avg if avg > 0 else 0
            lines.append((f"FPS: {fps:.1f}  Lat: {avg:.0f}ms", (255, 255, 255)))
        # Frame count
        lines.append((f"Frame: {self.frame_count}", (200, 200, 200)))
        # Draw background + text
        line_h = 22
        hud_h = len(lines) * line_h + 10
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (280, hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
        for i, (text, color) in enumerate(lines):
            y = 20 + i * line_h
            cv2.putText(frame_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)

    def _draw_status(self, frame_bgr, text, color=(0, 255, 255)):
        """Set status and draw HUD — legacy wrapper."""
        self._draw_hud(frame_bgr)

    def _send_frame(self, frame_bgr):
        with self.sock_lock:
            if self.sock is None:
                self._draw_status(frame_bgr, "NO CONN", (0, 0, 255))
                self._detect_status = 'NOCONN'
                return frame_bgr
            try:
                h, w = frame_bgr.shape[:2]
                wh = struct.pack("<HH", w, h)
                data = wh + frame_bgr.tobytes()
                hdr = struct.pack("<BI", 0, len(data))
                self.sock.sendall(hdr + data)
                result_len = struct.unpack("<I", self._recv_exact(4))[0]
                if result_len == 0:
                    cv2.line(frame_bgr, (w//2-20, h//2-20), (w//2+20, h//2+20), (0, 0, 255), 2)
                    cv2.line(frame_bgr, (w//2+20, h//2-20), (w//2-20, h//2+20), (0, 0, 255), 2)
                    self._draw_status(frame_bgr, "NO FACE", (0, 0, 255))
                    self._detect_status = 'FAIL'
                    return frame_bgr
                payload = self._recv_exact(result_len)
                expected_len = 8 + self.size * self.size * 3
                if result_len >= expected_len:
                    cx1, cy1, cx2, cy2 = struct.unpack("<hhhh", payload[:8])
                    rendered = np.frombuffer(payload[8:expected_len], dtype=np.uint8).reshape(self.size, self.size, 3)
                    cv2.rectangle(frame_bgr, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                    self._detect_status = 'OK'
                    frame_bgr = self._composite_face(frame_bgr, rendered, cx1, cy1, cx2, cy2)
                    return frame_bgr
                self._draw_status(frame_bgr, f"SIZE {result_len}!={expected_len}", (0, 200, 255))
                self._detect_status = 'SIZE'
                return frame_bgr
            except Exception as e:
                self._draw_status(frame_bgr, f"ERR: {str(e)[:40]}", (0, 0, 255))
                self.sock = None
                self._detect_status = 'ERR'
                return frame_bgr

    def _composite_face(self, frame, rendered, cx1, cy1, cx2, cy2):
        """Apply Wav2Lip mouth changes to original sharp face using delta blending.
        Uses an elliptical mask with wide, smooth feathering to avoid visible edges."""
        H, W = frame.shape[:2]
        cx1, cx2 = max(0, cx1), min(W, cx2)
        cy1, cy2 = max(0, cy1), min(H, cy2)
        crop_h, crop_w = cy2 - cy1, cx2 - cx1
        if crop_w < 5 or crop_h < 5:
            return frame
        orig_crop = frame[cy1:cy2, cx1:cx2].astype(np.float32)
        # Compute delta at model resolution
        orig_resized = cv2.resize(orig_crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        delta = rendered.astype(np.float32) - orig_resized
        half = self.size // 2
        mouth_delta = np.abs(delta[half:, :, :]).mean()
        self._mouth_delta = mouth_delta
        delta_up = cv2.resize(delta, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        enhanced = orig_crop + delta_up
        # Mouth-focused mask: fades in at nose, fades out at chin, spares cheeks
        yy = np.arange(crop_h, dtype=np.float32).reshape(-1, 1)
        mouth_center = crop_h * 0.62
        mouth_half = crop_h * 0.10
        mask = np.exp(-0.5 * ((yy - mouth_center) / mouth_half) ** 2)
        # Also narrow horizontally (Gaussian window) to spare cheeks
        xx = np.arange(crop_w, dtype=np.float32).reshape(1, -1)
        mask_h = np.exp(-0.5 * ((xx - crop_w / 2) / (crop_w * 0.28)) ** 2)
        mask = mask * mask_h
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0)
        mask = np.clip(mask, 0, 1)
        blended = enhanced * mask[..., None] + orig_crop * (1 - mask[..., None])
        frame[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame

    def _send_audio(self, audio_np):
        with self.sock_lock:
            if self.sock is None:
                return
            try:
                data = audio_np.astype(np.float32).tobytes()
                hdr = struct.pack("<BI", 1, len(data))
                self.sock.sendall(hdr + data)
                if self.frame_count % 30 == 0:
                    print(f"\r  [AUDIO SEND] {len(audio_np)} samples, max={audio_np.max():.3f}", end="")
            except Exception:
                self.sock = None

    def _send_reset(self):
        with self.sock_lock:
            if self.sock is None:
                return
            try:
                self.sock.sendall(struct.pack("<BI", 2, 0))
            except Exception:
                self.sock = None

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 65536))
            if not chunk:
                raise ConnectionError()
            buf.extend(chunk)
        return bytes(buf)

    def _setup_obs_window(self):
        """创建OBS可捕获的干净窗口 (无边框, 置顶, 可指定位置)"""
        import ctypes
        self._obs_win_name = "HeyGem_OBS"
        cv2.namedWindow(self._obs_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._obs_win_name, self.width, self.height)
        # 窗口置顶
        hwnd = ctypes.windll.user32.FindWindowW(None, self._obs_win_name)
        if hwnd:
            GWL_STYLE = -16
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            style &= ~(WS_CAPTION | WS_THICKFRAME)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                              SWP_NOMOVE | SWP_NOSIZE)
            print(f"OBS窗口已创建: 无边框, 置顶 ({self.width}x{self.height})")

    def _capture_loop(self, cap, frame_queue, video_fps=None):
        """通用采集线程: 摄像头或视频文件"""
        frame_delay = 1.0 / video_fps if video_fps else 0
        last_t = time.perf_counter()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.video_path and self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                elif self.video_path:
                    break  # 视频结束
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (self.width, self.height))

            # 视频模式保持原始帧率
            if frame_delay > 0:
                target_t = last_t + frame_delay
                now = time.perf_counter()
                if now < target_t:
                    time.sleep(target_t - now)
                last_t = time.perf_counter()

            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _capture_audio_from_video(self, audio_queue):
        """从视频文件提取音频，加载到流式缓冲区"""
        try:
            import tempfile, subprocess, os as _os
            probe = subprocess.run(
                f'ffprobe -v error -select_streams a:0 -show_entries stream=codec_type '
                f'-of default=noprint_wrappers=1:nokey=1 "{self.video_path}"',
                shell=True, capture_output=True, timeout=10,
            )
            if b"audio" not in probe.stdout:
                print("  视频无音频流，跳过音频")
                return
            audio_path = _os.path.join(tempfile.gettempdir(), "_heygem_audio.wav")
            subprocess.run(
                f'ffmpeg -y -i "{self.video_path}" -ac 1 -ar {AUDIO_SAMPLE_RATE} '
                f'-t 60 "{audio_path}"',
                shell=True, capture_output=True, timeout=30,
            )
            if not _os.path.exists(audio_path) or _os.path.getsize(audio_path) == 0:
                print("  音频提取失败: 输出文件为空")
                return
            import librosa
            audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
            # 加载到流式缓冲区（与 _load_external_audio 一致）
            self._external_audio = audio.astype(np.float32)
            self._audio_pos = 0
            self._audio_play_pending = True
            print(f"  视频音频已提取: {len(self._external_audio)/sr:.0f}s (流式发送)")
            _os.remove(audio_path)
        except Exception as e:
            print(f"  音频提取失败: {e}")

    def _load_external_audio(self, audio_queue):
        """加载外部音频文件 (mp3/wav), 重采样到16kHz, 分块流式发送"""
        try:
            import tempfile, subprocess, os as _os
            # 相对路径解析到脚本目录
            audio_path = self.audio_path
            if not _os.path.isabs(audio_path):
                audio_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), audio_path)
            if not _os.path.exists(audio_path):
                print(f"  [错误] 音频文件不存在: {audio_path}")
                return
            audio_wav = _os.path.join(tempfile.gettempdir(), "_heygem_ext_audio.wav")
            subprocess.run(
                f'ffmpeg -y -i "{audio_path}" -ac 1 -ar {AUDIO_SAMPLE_RATE} '
                f'"{audio_wav}"',
                shell=True, capture_output=True, timeout=30,
            )
            if not _os.path.exists(audio_wav) or _os.path.getsize(audio_wav) == 0:
                print("  音频转换失败")
                return
            import librosa
            audio, sr = librosa.load(audio_wav, sr=AUDIO_SAMPLE_RATE, mono=True)
            self._external_audio = audio.astype(np.float32)
            self._audio_pos = 0
            self._audio_start_time = None
            self._audio_play_pending = True  # 等管线预热后再播放
            print(f"  外部音频已加载: {len(self._external_audio)/sr:.0f}s (流式发送, 延迟播放)")
            _os.remove(audio_wav)
        except Exception as e:
            print(f"  外部音频加载失败: {e}")

    def _capture_mic(self, audio_queue):
        """麦克风采集"""
        try:
            import sounddevice as sd
        except ImportError:
            print("  [错误] sounddevice 未安装")
            return
        try:
            devices = sd.query_devices()
            mic_id = None
            # DirectSound BOYA 设备 (共享模式, 不冲突OBS)
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and 'BOYA' in d['name'] and 'DirectSound' in str(d):
                    mic_id = i
                    break
            if mic_id is None:
                # 任意BOYA
                for i, d in enumerate(devices):
                    if d['max_input_channels'] > 0 and 'BOYA' in d['name']:
                        mic_id = i
                        break
            if mic_id is None:
                mic_id = sd.default.device[0]
            dev_info = sd.query_devices(mic_id)
            dev_sr = int(dev_info['default_samplerate'])
            print(f"  麦克风: [{mic_id}] {dev_info['name']} {dev_sr}Hz")
            gain = 8.0
            import librosa
            chunk_count = 0
            while self.running:
                chunk = sd.rec(int(dev_sr * AUDIO_CHUNK_SECONDS),
                              samplerate=dev_sr, channels=1,
                              dtype="float32", blocking=True, device=mic_id)
                # 重采样到16kHz + 增益
                chunk_16k = librosa.resample(chunk.flatten(), orig_sr=dev_sr, target_sr=AUDIO_SAMPLE_RATE)
                chunk_16k = np.clip(chunk_16k * gain, -1.0, 1.0).astype(np.float32)
                level_max = float(np.abs(chunk_16k).max())
                level_std = float(chunk_16k.std())
                chunk_count += 1
                if chunk_count % 3 == 0:
                    print(f"\r  [MIC] max={level_max:.4f} std={level_std:.4f} {'<<<' if level_max > 0.05 else '(silent)'}  ", end="")
                try:
                    audio_queue.put_nowait(chunk_16k)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"  [麦克风错误] {e}")

    def _process_loop(self, frame_queue, audio_queue, cam_output):
        reconnect_interval = 100
        self._connect()

        while self.running:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                if (self.video_path or self.audio_path) and frame_queue.qsize() == 0 and not self.loop:
                    print("\n视频播放完毕")
                    self.running = False
                    break
                continue

            t0 = time.perf_counter()

            # 流式音频: 视频帧号驱动, 与视频精确同步
            has_stream_audio = hasattr(self, '_external_audio') and self._external_audio is not None
            if has_stream_audio:
                if self._audio_pos >= len(self._external_audio) and self.loop:
                    self._audio_pos = 0
                    self._send_reset()
                    if self.frame_count % 150 == 0:
                        print("\r  音频循环...", end="")
                fps = getattr(self, '_video_fps', 25) or 25
                target_pos = int(self.frame_count / fps * AUDIO_SAMPLE_RATE)
                if getattr(self, '_audio_play_pending', False) and self.frame_count >= 3:
                    try:
                        import sounddevice as sd
                        start_sample = int(self.frame_count / fps * AUDIO_SAMPLE_RATE)
                        sd.play(self._external_audio[start_sample:], samplerate=AUDIO_SAMPLE_RATE)
                    except Exception:
                        pass
                    self._audio_play_pending = False
                if target_pos > self._audio_pos and self._audio_pos < len(self._external_audio):
                    chunk_end = min(target_pos, len(self._external_audio))
                    chunk = self._external_audio[self._audio_pos:chunk_end]
                    if len(chunk) > 0:
                        self._send_audio(chunk)
                        if self.frame_count % 90 == 0:
                            print(f"\r  音频流: pos={self._audio_pos}/{len(self._external_audio)} chunk={len(chunk)}", end="")
                    self._audio_pos = chunk_end

            # 麦克风模式下持续发送音频
            if self.use_mic or not self.video_path:
                audio_parts = []
                while not audio_queue.empty():
                    try:
                        audio_parts.append(audio_queue.get_nowait())
                    except queue.Empty:
                        break
                if audio_parts:
                    combined = np.concatenate(audio_parts)
                    if self.frame_count % 15 == 0:
                        print(f"\r  发送音频: {len(combined)}样点, std={combined.std():.4f} sock={'OK' if self.sock else 'NO'}  ", end="")
                    self._send_audio(combined)
                elif self.frame_count % 90 == 0:
                    print(f"\r  音频队列空, mic={'OK' if self.use_mic else 'OFF'}  ", end="")

            # 发送帧
            if self.sock is not None:
                frame = self._send_frame(frame)
            else:
                self._detect_status = 'NOCONN'
                if self.frame_count % reconnect_interval == 0:
                    self._connect()

            dt = (time.perf_counter() - t0) * 1000
            self.latency_history.append(dt)
            if len(self.latency_history) > 50:
                self.latency_history.pop(0)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                avg = np.mean(self.latency_history)
                md = getattr(self, '_mouth_delta', 0)
                print(f"\rFPS: {1000/avg:.1f} | 延迟: {avg:.0f}ms | 人脸: {self._detect_status} | 嘴Δ: {md:.1f}/255 | #{self.frame_count}", end="")

            self._draw_hud(frame)
            if cam_output is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam_output.send(rgb)
                except Exception as e:
                    print(f"\r[虚拟摄像头错误] {e}", end="")
                    cam_output = None
            elif self.obs_win:
                cv2.imshow(self._obs_win_name, frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    self.running = False
            else:
                cv2.imshow("HeyGem Live (TCP)", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    self.running = False

    def start(self):
        print("=" * 50)
        print(f"HeyGem Live TCP Client ({self.size}×{self.size})")
        print("=" * 50)
        print(f"Server: {WSL_HOST}:{WSL_PORT}")

        # 打开视频或摄像头
        if self.video_path:
            if not os.path.exists(self.video_path):
                print(f"ERROR: 视频文件不存在: {self.video_path}")
                return
            cap = cv2.VideoCapture(self.video_path)
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"视频: {os.path.basename(self.video_path)} ({total_frames}帧 @ {fps_video:.0f}fps, {vw}x{vh})")
        else:
            cap = cv2.VideoCapture(CAMERA_ID)
            if not cap.isOpened():
                print("ERROR: 无法打开摄像头")
                return
            fps_video = None
            vw, vh = WIDTH, HEIGHT
            print(f"摄像头 #{CAMERA_ID}")

        # 输出分辨率: 竖屏自动切换或 --portrait 强制
        if self.portrait or (self.video_path and vh > vw):
            self.width, self.height = 720, 1280
        else:
            self.width, self.height = WIDTH, HEIGHT
        self._video_fps = fps_video
        print(f"输出: {self.width}x{self.height}")

        # 虚拟摄像头
        cam_output = None
        if self.use_virtualcam:
            try:
                import pyvirtualcam
                out_fps = int(fps_video) if fps_video else 25
                cam_output = pyvirtualcam.Camera(width=self.width, height=self.height, fps=out_fps,
                                                  fmt=pyvirtualcam.PixelFormat.RGB)
                print(f"虚拟摄像头: {self.width}x{self.height} @{out_fps}fps")
            except ImportError:
                print("pyvirtualcam 未安装，使用预览窗口")
            except Exception as e:
                print(f"虚拟摄像头错误: {e}")

        # 音频
        if self.use_mic:
            audio_ok = True
            print(f"麦克风: {AUDIO_SAMPLE_RATE}Hz (嘴型驱动)")
        elif self.audio_path:
            audio_ok = True
            print(f"外部音频: {os.path.basename(self.audio_path)}")
        elif self.video_path:
            audio_ok = True  # 从视频提取
        else:
            audio_ok = False
            try:
                import sounddevice
                print(f"麦克风: {AUDIO_SAMPLE_RATE}Hz")
                audio_ok = True
            except ImportError:
                print("麦克风: 不可用")

        # 启动线程
        self.running = True
        frame_queue = queue.Queue(maxsize=2)
        mic_mode = self.use_mic or not self.video_path
        audio_queue = queue.Queue(maxsize=20 if mic_mode else 1)

        threading.Thread(target=self._capture_loop,
                         args=(cap, frame_queue, fps_video),
                         daemon=True).start()

        if self.use_mic:
            threading.Thread(target=self._capture_mic,
                             args=(audio_queue,), daemon=True).start()
        elif self.audio_path:
            threading.Thread(target=self._load_external_audio,
                             args=(audio_queue,), daemon=True).start()
        elif self.video_path:
            threading.Thread(target=self._capture_audio_from_video,
                             args=(audio_queue,), daemon=True).start()
        elif audio_ok:
            threading.Thread(target=self._capture_mic,
                             args=(audio_queue,), daemon=True).start()

        # 清理残留窗口 + 设置预览窗口 (可自由调整大小)
        cv2.destroyAllWindows()
        if self.obs_win:
            self._setup_obs_window()
        elif not self.use_virtualcam:
            cv2.namedWindow("HeyGem Live (TCP)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HeyGem Live (TCP)", self.width, self.height)
            cv2.moveWindow("HeyGem Live (TCP)", 0, 0)

        mode = "视频文件" if self.video_path else "摄像头"
        print(f"\n{mode}模式运行中... Ctrl+C 停止, Q 退出预览\n")
        try:
            self._process_loop(frame_queue, audio_queue, cam_output)
        except KeyboardInterrupt:
            print("\n停止中...")

        self.running = False
        if self.sock:
            self.sock.close()
        cap.release()
        if cam_output:
            cam_output.close()
        cv2.destroyAllWindows()
        print("客户端已停止")


if __name__ == "__main__":
    # Single-instance check
    import ctypes, sys as _sys
    _mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\HeyGemTCPClient")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        print("HeyGem TCP 客户端已在运行中，请先关闭现有窗口")
        _sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="视频文件路径")
    parser.add_argument("--virtualcam", action="store_true", help="OBS虚拟摄像头输出")
    parser.add_argument("--loop", action="store_true", help="视频循环播放")
    parser.add_argument("--mic", action="store_true", help="麦克风驱动嘴型(可与视频组合)")
    parser.add_argument("--portrait", action="store_true", help="竖屏输出 720x1280")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--audio", type=str, default=None, help="独立音频文件路径")
    parser.add_argument("--size", type=int, default=96, choices=[96, 256], help="模型分辨率 (默认96)")
    args = parser.parse_args()
    if args.video:
        CAMERA_ID = None
    else:
        CAMERA_ID = args.camera
    TCPStreamingClient(video_path=args.video, use_virtualcam=args.virtualcam,
                       loop=args.loop, use_mic=args.mic,
                       portrait=args.portrait, audio_path=args.audio,
                       size=args.size).start()
