"""Final end-to-end TCP test: server+client on same machine, real audio, measure output."""
import sys, os, time, struct, socket, threading
import numpy as np
import cv2
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))

PORT = 7865

# Load real audio
audio_full, sr = sf.read("1004_format.wav")
if audio_full.ndim > 1:
    audio_full = audio_full[:, 0]
audio_full = audio_full.astype(np.float32)
if sr != 16000:
    import librosa
    audio_full = librosa.resample(audio_full, orig_sr=sr, target_sr=16000)
    sr = 16000
print(f"Audio: {len(audio_full)/sr:.1f}s, RMS={np.sqrt(np.mean(audio_full**2)):.3f}")

# Load video
cap = cv2.VideoCapture("1004-r.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
print(f"Video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames @ {fps} fps")

def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("connection closed")
        buf.extend(chunk)
    return bytes(buf)

def run_tcp_test(size, use_align):
    from pipeline_wav2lip import StreamingPipeline

    mode = "ALIGNED" if use_align else "NO-ALIGN"
    print(f"\n{'='*60}")
    print(f"TCP Test: {size}×{size} — {mode}")
    print(f"{'='*60}")

    pipeline = StreamingPipeline(detect_interval=2, test_audio=False, size=size, use_align=use_align)
    pipeline.start()

    # Start TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", PORT))
    server.listen(1)
    server.settimeout(10)

    results = {"frames": 0, "faces": 0, "mouth_deltas": [], "errors": []}

    def handle():
        try:
            conn, addr = server.accept()
            conn.settimeout(10)
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

                elif msg_type == 0:  # Frame
                    if payload_len < 4:
                        continue
                    w = struct.unpack("<H", payload[0:2])[0]
                    h = struct.unpack("<H", payload[2:4])[0]
                    frame_data = payload[4:]
                    if len(frame_data) != w * h * 3:
                        continue
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(h, w, 3).copy()
                    results["frames"] += 1

                    rendered, coords = pipeline.process_frame(frame)
                    if rendered is not None:
                        results["faces"] += 1
                        cx1, cy1, cx2, cy2 = coords
                        coord_bytes = struct.pack("<hhhh", cx1, cy1, cx2, cy2)
                        face_bytes = rendered.tobytes()
                        payload_out = coord_bytes + face_bytes
                        conn.sendall(struct.pack("<I", len(payload_out)) + payload_out)

                        # Measure mouth delta
                        cx1_c = max(0, cx1)
                        cx2_c = min(w, cx2)
                        cy1_c = max(0, cy1)
                        cy2_c = min(h, cy2)
                        crop_h = cy2_c - cy1_c
                        crop_w = cx2_c - cx1_c
                        if crop_w >= 5 and crop_h >= 5:
                            orig_crop = frame[cy1_c:cy2_c, cx1_c:cx2_c].astype(np.float32)
                            orig_resized = cv2.resize(orig_crop, (size, size), interpolation=cv2.INTER_AREA)
                            delta = rendered.astype(np.float32) - orig_resized
                            half = 128 if size == 256 else 48
                            mouth_delta = np.abs(delta[half:, :, :]).mean()
                            results["mouth_deltas"].append(mouth_delta)
                    else:
                        conn.sendall(struct.pack("<I", 0))

        except (ConnectionError, ConnectionResetError, socket.timeout):
            pass
        except Exception as e:
            results["errors"].append(str(e))
        finally:
            try:
                conn.close()
            except:
                pass

    srv_thread = threading.Thread(target=handle, daemon=True)
    srv_thread.start()
    time.sleep(0.5)

    # Client
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(10)
    client.connect(("127.0.0.1", PORT))
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    max_frames = 60
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Send audio
        a_start = int(i / fps * sr)
        a_end = int((i + 1) / fps * sr)
        if a_end <= len(audio_full):
            audio_chunk = audio_full[a_start:a_end]
        else:
            audio_chunk = np.zeros(int(sr/fps), dtype=np.float32)
        hdr = struct.pack("<BI", 1, len(audio_chunk) * 4)
        client.sendall(hdr + audio_chunk.astype(np.float32).tobytes())

        # Send frame
        h, w = frame.shape[:2]
        wh = struct.pack("<HH", w, h)
        data = wh + frame.tobytes()
        hdr = struct.pack("<BI", 0, len(data))
        client.sendall(hdr + data)

        # Receive result
        try:
            result_len = struct.unpack("<I", recv_exact(client, 4))[0]
        except:
            break
        if result_len > 0:
            payload = recv_exact(client, result_len)

    client.close()
    server.close()
    pipeline.stop()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Results
    print(f"  Frames sent: {results['frames']}")
    print(f"  Faces detected: {results['faces']}")
    if results["mouth_deltas"]:
        avg = np.mean(results["mouth_deltas"])
        print(f"  Avg mouth delta (model level): {avg:.2f}/255 = {avg/255*100:.2f}%")
        print(f"  Max mouth delta: {np.max(results['mouth_deltas']):.2f}/255")
    if results["errors"]:
        print(f"  Errors: {results['errors']}")
    return results

# Run all three configurations
r96 = run_tcp_test(96, False)
r256_noalign = run_tcp_test(256, False)
r256_align = run_tcp_test(256, True)

print(f"\n{'='*60}")
print(f"FINAL COMPARISON (TCP + Real Audio)")
print(f"{'='*60}")
for label, r in [("96 no-align", r96), ("256 no-align", r256_noalign), ("256 aligned", r256_align)]:
    if r["mouth_deltas"]:
        avg = np.mean(r["mouth_deltas"])
        print(f"  {label:20s}: delta={avg:.2f}/255 ({avg/255*100:.2f}%), faces={r['faces']}/{r['frames']}")
    else:
        print(f"  {label:20s}: NO MOUTH DELTAS, faces={r['faces']}/{r['frames']}")
print(f"{'='*60}")
print("DONE")
