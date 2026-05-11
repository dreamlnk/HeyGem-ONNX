"""TCP流式管线实时测试 (WSL侧)"""
import socket, struct, time, subprocess
import numpy as np
import cv2
import librosa


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        buf.extend(sock.recv(n - len(buf)))
    return bytes(buf)


def send_frame(sock, frame):
    data = frame.tobytes()
    hdr = struct.pack("<BI", 0, len(data))
    sock.sendall(hdr + data)
    result_len = struct.unpack("<I", recv_exact(sock, 4))[0]
    result = recv_exact(sock, result_len)
    return np.frombuffer(result, dtype=np.uint8).reshape(720, 1280, 3)


def send_audio(sock, audio_np):
    data = audio_np.astype(np.float32).tobytes()
    hdr = struct.pack("<BI", 1, len(data))
    sock.sendall(hdr + data)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15)
    sock.connect(("127.0.0.1", 7862))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("已连接 TCP 服务端")

    cap = cv2.VideoCapture("example/video.mp4")

    # 提取真实音频
    audio_path = "/tmp/live_test_audio.wav"
    subprocess.run(
        f"ffmpeg -y -i example/video.mp4 -ac 1 -ar 16000 -t 10 {audio_path}",
        shell=True, capture_output=True,
    )
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    send_audio(sock, audio)
    print(f"音频已发送: {len(audio)/sr:.1f}s")

    # Warmup
    print("预热...")
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    for _ in range(5):
        send_frame(sock, frame)
    print("预热完成")

    # 流式测试
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    N = 120
    print(f"\n流式处理 {N} 帧...")
    t_start = time.perf_counter()
    times = []

    for i in range(N):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))

        t0 = time.perf_counter()
        result = send_frame(sock, frame)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

        if (i + 1) % 30 == 0:
            elapsed = time.perf_counter() - t_start
            fps = (i + 1) / elapsed
            avg = np.mean(times[-30:])
            print(f"  帧 {i+1}/{N} | 延迟 {avg:.0f}ms | 实际 {fps:.1f} FPS")

    elapsed = time.perf_counter() - t_start
    times_all = np.array(times)
    stable = times_all[5:]

    print(f"\n===== 结果 =====")
    print(f"实际吞吐: {len(times)/elapsed:.1f} FPS")
    print(f"稳定延迟: avg={stable.mean():.0f}ms ({1000/stable.mean():.0f} FPS)")
    print(f"  min={stable.min():.0f}ms  p50={np.median(stable):.0f}ms  p99={np.percentile(stable,99):.0f}ms")

    cv2.imwrite("/tmp/tcp_live_result.jpg", result)
    print(f"末帧保存: /tmp/tcp_live_result.jpg")

    sock.close()
    cap.release()


if __name__ == "__main__":
    main()
