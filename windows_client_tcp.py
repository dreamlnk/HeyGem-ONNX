"""
Windows 客户端 (TCP版) - 原始BGR传输, 零编解码开销
运行: python windows_client_tcp.py [--virtualcam]
"""
import time, queue, struct, socket, threading, argparse
import numpy as np
import cv2

# === 配置 ===
WSL_HOST = "127.0.0.1"
WSL_PORT = 7862
WIDTH, HEIGHT = 1280, 720
FRAME_SIZE = WIDTH * HEIGHT * 3  # BGR
CAMERA_ID = 0
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 0.5


class TCPStreamingClient:
    def __init__(self, use_virtualcam=False):
        self.running = False
        self.use_virtualcam = use_virtualcam
        self.latency_history = []
        self.frame_count = 0
        self.sock = None
        self.sock_lock = threading.Lock()

    def _connect(self):
        """建立TCP连接 (带重试)"""
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
        """发送BGR帧, 返回渲染结果"""
        with self.sock_lock:
            if self.sock is None:
                return None
            try:
                data = frame_bgr.tobytes()
                hdr = struct.pack("<BI", 0, len(data))
                self.sock.sendall(hdr + data)

                # 接收结果
                result_len = struct.unpack("<I", self._recv_exact(4))[0]
                if result_len > 0 and result_len == FRAME_SIZE:
                    return np.frombuffer(self._recv_exact(result_len),
                                         dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            except Exception:
                self.sock = None
                return None

    def _send_audio(self, audio_np):
        """发送音频数据 (不等待回复)"""
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

    def _capture_webcam(self, cap, frame_queue):
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _capture_audio(self, audio_queue):
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

        while self.running:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            # 发送音频 (非阻塞)
            audio_parts = []
            while not audio_queue.empty():
                try:
                    audio_parts.append(audio_queue.get_nowait())
                except queue.Empty:
                    break
            if audio_parts:
                self._send_audio(np.concatenate(audio_parts))

            # 发送帧并接收结果
            if self.sock is not None:
                result = self._send_frame(frame)
                if result is not None:
                    frame = result
            else:
                # 重连
                if self.frame_count % reconnect_interval == 0:
                    self._connect()

            # 显示
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

        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        print(f"Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

        cam_output = None
        if self.use_virtualcam:
            try:
                import pyvirtualcam
                cam_output = pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=25,
                                                  fmt=pyvirtualcam.PixelFormat.RGB)
                print(f"Virtual cam: {WIDTH}x{HEIGHT}")
            except Exception as e:
                print(f"Virtual cam error: {e}")

        audio_ok = False
        try:
            import sounddevice
            print(f"Audio: {AUDIO_SAMPLE_RATE}Hz")
            audio_ok = True
        except ImportError:
            print("Audio: not available")

        self.running = True
        frame_queue = queue.Queue(maxsize=2)
        audio_queue = queue.Queue(maxsize=20)

        threading.Thread(target=self._capture_webcam, args=(cap, frame_queue),
                         daemon=True).start()
        if audio_ok:
            threading.Thread(target=self._capture_audio, args=(audio_queue,),
                             daemon=True).start()

        print("\nRunning! Ctrl+C to stop, Q to quit\n")
        try:
            self._process_loop(frame_queue, audio_queue, cam_output)
        except KeyboardInterrupt:
            print("\nShutting down...")

        self.running = False
        if self.sock:
            self.sock.close()
        cap.release()
        if cam_output:
            cam_output.close()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--virtualcam", action="store_true")
    args = parser.parse_args()
    TCPStreamingClient(use_virtualcam=args.virtualcam).start()
