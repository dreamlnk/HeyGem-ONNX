"""
流式管线服务端 (TCP版) - 原始BGR字节流, 无JPEG/HTTP开销
"""
import os, sys, time, struct, socket, threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from pipeline_wav2lip import StreamingPipeline

PORT = 7863

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_CONN = os.path.join(BASE_DIR, "debug_conn.log")
LOG_MAIN = os.path.join(BASE_DIR, "debug_main.log")


def recv_exact(sock, n):
    """接收精确n字节"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def handle_client(conn, addr, pipeline):
    """Handle a single client connection."""
    with open(LOG_CONN, "a") as f:
        f.write(f"CONNECTED {addr}\n")
    pipeline.last_bbox = None
    pipeline.last_kps = None
    try:
        while True:
            hdr = recv_exact(conn, 5)
            msg_type = hdr[0]
            payload_len = struct.unpack("<I", hdr[1:5])[0]

            if payload_len > 10_000_000:
                break

            payload = recv_exact(conn, payload_len)

            if msg_type == 1:  # Audio
                audio_np = np.frombuffer(payload, dtype=np.float32).copy()
                if len(audio_np) > 0:
                    pipeline.feed_audio(audio_np)
                    with open(LOG_CONN, "a") as f:
                        f.write(f"AUDIO {len(audio_np)} max={audio_np.max():.3f}\n")
                continue

            elif msg_type == 2:  # 重置
                pipeline.last_bbox = None
                pipeline.last_kps = None
                pipeline._bbox_history.clear()
                if hasattr(pipeline, '_kps_history'):
                    pipeline._kps_history.clear()
                with pipeline.audio_lock:
                    pipeline.audio_buffer = np.array([], dtype=np.float32)
                continue

            elif msg_type == 0:  # Frame
                if payload_len < 4:
                    continue
                w = struct.unpack("<H", payload[0:2])[0]
                h = struct.unpack("<H", payload[2:4])[0]
                frame_data = payload[4:]
                expected = w * h * 3
                if len(frame_data) != expected:
                    continue

                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(h, w, 3).copy()
                with open(LOG_CONN, "a") as f:
                    f.write(f"FRAME {pipeline.frame_idx} {w}x{h}\n")
                rendered_96, coords = pipeline.process_frame(frame)
                if rendered_96 is not None:
                    cx1, cy1, cx2, cy2 = coords
                    coord_bytes = struct.pack("<hhhh", cx1, cy1, cx2, cy2)
                    face_bytes = rendered_96.tobytes()
                    payload = coord_bytes + face_bytes
                    conn.sendall(struct.pack("<I", len(payload)) + payload)
                else:
                    conn.sendall(struct.pack("<I", 0))

    except (ConnectionError, ConnectionResetError, struct.error):
        pass
    except Exception as e:
        import traceback as _tb
        print(f"[错误] {addr}: {e}")
        _tb.print_exc()
    finally:
        conn.close()
        print(f"[断开] {addr}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-audio", action="store_true", help="使用振荡测试音频验证唇形")
    parser.add_argument("--size", type=int, default=96, choices=[96, 256], help="模型分辨率 (默认96)")
    args = parser.parse_args()

    with open(LOG_MAIN, "w") as f:
        f.write(f"Server starting, PID: {os.getpid()}\n")
        f.write(f"CWD: {os.getcwd()}\n")
        f.write(f"Script: {__file__}\n")
        f.write(f"Size: {args.size}\n")
    print("=" * 60, flush=True)
    print(f"HeyGem TCP ({args.size}×{args.size})", flush=True)
    print("=" * 60, flush=True)

    print(f"Loading pipeline ({args.size}×{args.size})...", flush=True)
    pipeline = StreamingPipeline(detect_interval=2, test_audio=args.test_audio, size=args.size)
    pipeline.start()
    if args.test_audio:
        pipeline.feed_test_audio()
    print(f"管线就绪 (test_audio={args.test_audio})", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", PORT))
    sock.listen(1)
    print(f"监听端口 {PORT}...", flush=True)

    try:
        while True:
            with open(LOG_CONN, "a") as f:
                f.write("Waiting for accept...\n")
            conn, addr = sock.accept()
            with open(LOG_CONN, "a") as f:
                f.write(f"ACCEPT {addr}\n")
            t = threading.Thread(target=handle_client, args=(conn, addr, pipeline),
                                 daemon=True)
            t.start()
            t.join()  # Wait for client to disconnect before accepting next
    except KeyboardInterrupt:
        print("\n停止服务")
    finally:
        sock.close()
        pipeline.stop()


if __name__ == "__main__":
    main()
