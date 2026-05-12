"""
流式管线服务端 (TCP版) - 原始BGR字节流, 无JPEG/HTTP开销
"""
import os, sys, time, struct, socket, threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_complete import StreamingPipeline

PORT = 7862


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
    """处理单个客户端连接"""
    print(f"[连接] {addr}")
    try:
        while True:
            # 读取类型标记
            hdr = recv_exact(conn, 5)
            msg_type = hdr[0]
            payload_len = struct.unpack("<I", hdr[1:5])[0]

            if payload_len > 10_000_000:  # 安全上限 ~10MB
                break

            payload = recv_exact(conn, payload_len)

            if msg_type == 1:  # 音频
                audio_np = np.frombuffer(payload, dtype=np.float32).copy()
                if len(audio_np) > 0:
                    pipeline.feed_audio(audio_np)
                # 音频不回复
                continue

            elif msg_type == 2:  # 重置
                pipeline.source_face_tensor = None
                pipeline.last_bbox = None
                pipeline.last_kps = None
                continue

            elif msg_type == 0:  # 帧
                # 新协议: [type=0][2B width LE][2B height LE][payload]
                # payload_len = width * height * 3
                if payload_len < 4:
                    continue
                w = struct.unpack("<H", payload[0:2])[0]
                h = struct.unpack("<H", payload[2:4])[0]
                frame_data = payload[4:]
                expected = w * h * 3
                if len(frame_data) != expected:
                    continue

                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(h, w, 3).copy()
                result = pipeline.process_frame(frame)
                result_bytes = result.tobytes()
                conn.sendall(struct.pack("<I", len(result_bytes)) + result_bytes)

    except (ConnectionError, ConnectionResetError, struct.error):
        pass
    except Exception as e:
        print(f"[错误] {addr}: {e}")
    finally:
        conn.close()
        print(f"[断开] {addr}")


def main():
    print("=" * 60)
    print("HeyGem TCP 流式服务端")
    print("=" * 60)

    print("初始化管线...")
    pipeline = StreamingPipeline(detect_interval=5)
    pipeline.start()
    # 预初始化音频特征 (避免首次帧跳过DINet)
    pipeline.latest_audio_feat = np.zeros((256, 256), dtype=np.float32)
    print("管线就绪")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", PORT))
    sock.listen(2)
    print(f"监听端口 {PORT}...")

    try:
        while True:
            conn, addr = sock.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr, pipeline),
                                 daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n停止服务")
    finally:
        sock.close()
        pipeline.stop()


if __name__ == "__main__":
    main()
