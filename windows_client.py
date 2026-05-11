"""
Windows 客户端: 摄像头 + 麦克风 -> WSL管线 -> OBS虚拟摄像头
运行: python windows_client.py [--preview|--virtualcam]
依赖: pip install opencv-python requests pyvirtualcam sounddevice numpy
"""
import time, queue, threading, argparse
import numpy as np
import cv2
import requests

# === 配置 ===
WSL_HOST = "http://localhost:7861"
CAMERA_ID = 0
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 0.5
JPEG_QUALITY = 80


class WindowsStreamingClient:
    def __init__(self, use_virtualcam=False):
        self.running = False
        self.use_virtualcam = use_virtualcam
        self.latency_history = []
        self.frame_count = 0

    def _capture_webcam(self, cap, frame_queue):
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _capture_audio(self, audio_queue):
        try:
            import sounddevice as sd
            while self.running:
                chunk = sd.rec(
                    int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS),
                    samplerate=AUDIO_SAMPLE_RATE, channels=1,
                    dtype="float32", blocking=True,
                )
                try:
                    audio_queue.put_nowait(chunk.flatten())
                except queue.Full:
                    pass
        except ImportError:
            pass  # sounddevice not available

    def _process_loop(self, frame_queue, audio_queue, cam_output):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2, pool_maxsize=4, max_retries=0
        )
        session.mount("http://", adapter)

        while self.running:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            # Encode and send
            try:
                _, jpeg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                files = {"frame": ("frame.jpg", jpeg.tobytes(), "image/jpeg")}

                # Gather audio chunks
                audio_parts = []
                while not audio_queue.empty():
                    try:
                        audio_parts.append(audio_queue.get_nowait())
                    except queue.Empty:
                        break
                if audio_parts:
                    audio_data = np.concatenate(audio_parts).astype(np.float32)
                    files["audio"] = ("audio.raw", audio_data.tobytes(),
                                      "application/octet-stream")

                resp = session.post(
                    f"{WSL_HOST}/process_frame",
                    files=files, timeout=2,
                )

                if resp.status_code == 200 and len(resp.content) > 1000:
                    result = cv2.imdecode(
                        np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR
                    )
                    if result is not None and result.shape == frame.shape:
                        frame = result

            except requests.exceptions.Timeout:
                pass
            except requests.exceptions.ConnectionError:
                print(f"\n[WARN] WSL connection lost", end="")
            except Exception as e:
                print(f"\n[ERR] {e}", end="")

            # Track latency
            dt = (time.perf_counter() - t0) * 1000
            self.latency_history.append(dt)
            if len(self.latency_history) > 50:
                self.latency_history.pop(0)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                avg = np.mean(self.latency_history)
                print(f"\rFPS: {1000/avg:.1f} | 延迟: {avg:.0f}ms | #{self.frame_count}", end="")

            # Output
            if cam_output is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_output.send(rgb)
            else:
                cv2.imshow("HeyGem Live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False

    def start(self):
        print("=" * 50)
        print("HeyGem Live Streaming Client")
        print("=" * 50)

        # Check server
        print(f"WSL server: {WSL_HOST}")
        try:
            r = requests.get(f"{WSL_HOST}/health", timeout=5)
            print(f"  Status: {r.json().get('status', '?')}")
        except Exception:
            print("  [WARN] Server not responding, will retry...")

        # Camera
        print(f"Camera #{CAMERA_ID}...")
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_HEIGHT)
        print(f"  Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

        # Virtual camera
        cam_output = None
        if self.use_virtualcam:
            try:
                import pyvirtualcam
                cam_output = pyvirtualcam.Camera(
                    width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT, fps=25,
                    fmt=pyvirtualcam.PixelFormat.RGB,
                )
                print(f"  Virtual camera: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} @25fps")
            except ImportError:
                print("  pyvirtualcam not installed, using preview window")
            except Exception as e:
                print(f"  Virtual camera error: {e}")

        # Audio
        audio_ok = False
        try:
            import sounddevice as sd
            print(f"  Audio: {AUDIO_SAMPLE_RATE}Hz")
            audio_ok = True
        except ImportError:
            print("  Audio: sounddevice not installed (no audio)")

        # Start threads
        self.running = True
        frame_queue = queue.Queue(maxsize=2)
        audio_queue = queue.Queue(maxsize=20)

        threading.Thread(target=self._capture_webcam, args=(cap, frame_queue),
                         daemon=True).start()
        if audio_ok:
            threading.Thread(target=self._capture_audio, args=(audio_queue,),
                             daemon=True).start()

        print("\nRunning! Press Ctrl+C to stop, Q to quit preview\n")
        try:
            self._process_loop(frame_queue, audio_queue, cam_output)
        except KeyboardInterrupt:
            print("\nShutting down...")

        self.running = False
        cap.release()
        if cam_output is not None:
            cam_output.close()
        cv2.destroyAllWindows()
        print("Client stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--virtualcam", action="store_true", help="Use OBS virtual camera output")
    parser.add_argument("--preview", action="store_true", default=True, help="Use preview window (default)")
    args = parser.parse_args()
    client = WindowsStreamingClient(use_virtualcam=args.virtualcam)
    client.start()
