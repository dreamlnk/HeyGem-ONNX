"""
Windows 客户端 (TCP版) - 原始BGR传输, 零编解码开销
运行:
  python windows_client_tcp.py                      # 摄像头预览
  python windows_client_tcp.py --virtualcam          # 摄像头+OBS虚拟摄像头
  python windows_client_tcp.py --video example/video.mp4           # 视频文件预览
  python windows_client_tcp.py --video example/video.mp4 --virtualcam  # 视频+OBS
"""
import time, queue, struct, socket, threading, argparse, os
import numpy as np
import cv2

# === 配置 ===
WSL_HOST = "127.0.0.1"
WSL_PORT = 7862
WIDTH, HEIGHT = 1280, 720
FRAME_SIZE = WIDTH * HEIGHT * 3
CAMERA_ID = 0
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 0.5


class TCPStreamingClient:
    def __init__(self, video_path=None, use_virtualcam=False, loop=False):
        self.running = False
        self.video_path = video_path
        self.loop = loop
        self.use_virtualcam = use_virtualcam
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

    def _send_frame(self, frame_bgr):
        with self.sock_lock:
            if self.sock is None:
                return None
            try:
                data = frame_bgr.tobytes()
                hdr = struct.pack("<BI", 0, len(data))
                self.sock.sendall(hdr + data)
                result_len = struct.unpack("<I", self._recv_exact(4))[0]
                if result_len > 0 and result_len == FRAME_SIZE:
                    return np.frombuffer(self._recv_exact(result_len),
                                         dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            except Exception:
                self.sock = None
                return None

    def _send_audio(self, audio_np):
        with self.sock_lock:
            if self.sock is None:
                return
            try:
                data = audio_np.astype(np.float32).tobytes()
                hdr = struct.pack("<BI", 1, len(data))
                self.sock.sendall(hdr + data)
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

            frame = cv2.resize(frame, (WIDTH, HEIGHT))

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
        """从视频文件提取音频并发送"""
        try:
            audio_path = os.path.join(os.path.dirname(self.video_path),
                                      "_temp_audio.wav")
            import subprocess
            subprocess.run(
                f'ffmpeg -y -i "{self.video_path}" -ac 1 -ar {AUDIO_SAMPLE_RATE} '
                f'-t 60 "{audio_path}"',
                shell=True, capture_output=True,
            )
            import librosa
            audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
            # 一次性发送全部音频
            audio_queue.put(audio.astype(np.float32))
            print(f"  视频音频已提取: {len(audio)/sr:.0f}s")
            os.remove(audio_path)
        except Exception as e:
            print(f"  音频提取失败: {e}")

    def _capture_mic(self, audio_queue):
        """麦克风采集"""
        try:
            import sounddevice as sd
            while self.running:
                chunk = sd.rec(int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS),
                               samplerate=AUDIO_SAMPLE_RATE, channels=1,
                               dtype="float32", blocking=True)
                try:
                    audio_queue.put_nowait(chunk.flatten())
                except queue.Full:
                    pass
        except ImportError:
            pass

    def _process_loop(self, frame_queue, audio_queue, cam_output):
        reconnect_interval = 100
        self._connect()

        # 视频模式下,一次性发送全部音频
        if self.video_path and not audio_queue.empty():
            audio_data = audio_queue.get()
            self._send_audio(audio_data)

        while self.running:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                if self.video_path and frame_queue.qsize() == 0 and not self.loop:
                    print("\n视频播放完毕")
                    self.running = False
                    break
                continue

            t0 = time.perf_counter()

            # 麦克风模式下持续发送音频
            if not self.video_path:
                audio_parts = []
                while not audio_queue.empty():
                    try:
                        audio_parts.append(audio_queue.get_nowait())
                    except queue.Empty:
                        break
                if audio_parts:
                    self._send_audio(np.concatenate(audio_parts))

            # 发送帧
            if self.sock is not None:
                result = self._send_frame(frame)
                if result is not None:
                    frame = result
            else:
                if self.frame_count % reconnect_interval == 0:
                    self._connect()

            dt = (time.perf_counter() - t0) * 1000
            self.latency_history.append(dt)
            if len(self.latency_history) > 50:
                self.latency_history.pop(0)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                avg = np.mean(self.latency_history)
                print(f"\rFPS: {1000/avg:.1f} | 延迟: {avg:.0f}ms | #{self.frame_count}", end="")

            if cam_output is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_output.send(rgb)
            else:
                cv2.imshow("HeyGem Live (TCP)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False

    def start(self):
        print("=" * 50)
        print("HeyGem Live TCP Client")
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
            print(f"视频: {os.path.basename(self.video_path)} ({total_frames}帧 @ {fps_video:.0f}fps)")
        else:
            cap = cv2.VideoCapture(CAMERA_ID)
            if not cap.isOpened():
                print("ERROR: 无法打开摄像头")
                return
            fps_video = None
            print(f"摄像头 #{CAMERA_ID}")

        print(f"分辨率: {WIDTH}x{HEIGHT}")

        # 虚拟摄像头
        cam_output = None
        if self.use_virtualcam:
            try:
                import pyvirtualcam
                out_fps = int(fps_video) if fps_video else 25
                cam_output = pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=out_fps,
                                                  fmt=pyvirtualcam.PixelFormat.RGB)
                print(f"虚拟摄像头: {WIDTH}x{HEIGHT} @{out_fps}fps")
            except ImportError:
                print("pyvirtualcam 未安装，使用预览窗口")
            except Exception as e:
                print(f"虚拟摄像头错误: {e}")

        # 音频
        if self.video_path:
            audio_ok = True
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
        audio_queue = queue.Queue(maxsize=20 if not self.video_path else 1)

        threading.Thread(target=self._capture_loop,
                         args=(cap, frame_queue, fps_video),
                         daemon=True).start()

        if self.video_path:
            threading.Thread(target=self._capture_audio_from_video,
                             args=(audio_queue,), daemon=True).start()
        elif audio_ok:
            threading.Thread(target=self._capture_mic,
                             args=(audio_queue,), daemon=True).start()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="视频文件路径")
    parser.add_argument("--virtualcam", action="store_true", help="OBS虚拟摄像头输出")
    parser.add_argument("--loop", action="store_true", help="视频循环播放")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    args = parser.parse_args()
    if args.video:
        CAMERA_ID = None
    else:
        CAMERA_ID = args.camera
    TCPStreamingClient(video_path=args.video, use_virtualcam=args.virtualcam,
                       loop=args.loop).start()
