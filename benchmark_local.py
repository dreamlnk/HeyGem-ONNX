"""
本地离线压测 — 在GPU服务器上用本地音视频跑全管线，输出合成视频 + 性能报告
用法: python benchmark_local.py --video test.mp4 --size 256 --output result.mp4
"""
import os, sys, time, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_wav2lip import StreamingPipeline


def load_audio(path, sr=16000):
    """加载音频文件 (支持 wav/mp3/flac 等)"""
    import librosa
    audio, fs = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def composite_face(frame, rendered, cx1, cy1, cx2, cy2, size):
    """与 windows_client_tcp._composite_face 完全一致的 delta 合成逻辑"""
    H, W = frame.shape[:2]
    cx1, cx2 = max(0, cx1), min(W, cx2)
    cy1, cy2 = max(0, cy1), min(H, cy2)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1
    if crop_w < 5 or crop_h < 5:
        return frame, 0.0

    orig_crop = frame[cy1:cy2, cx1:cx2].astype(np.float32)
    orig_resized = cv2.resize(orig_crop, (size, size), interpolation=cv2.INTER_AREA)
    delta = rendered.astype(np.float32) - orig_resized
    half = size // 2
    mouth_delta = np.abs(delta[half:, :, :]).mean()
    delta_up = cv2.resize(delta, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
    enhanced = orig_crop + delta_up

    yy = np.arange(crop_h, dtype=np.float32).reshape(-1, 1)
    mouth_center = crop_h * 0.65
    mouth_half = crop_h * 0.16
    mask = np.exp(-0.5 * ((yy - mouth_center) / mouth_half) ** 2)
    xx = np.arange(crop_w, dtype=np.float32).reshape(1, -1)
    mask_h = np.exp(-0.5 * ((xx - crop_w / 2) / (crop_w * 0.42)) ** 2)
    mask = mask * mask_h
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5.0)
    mask = np.clip(mask, 0, 1)
    blended = enhanced * mask[..., None] + orig_crop * (1 - mask[..., None])
    frame[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame, mouth_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--audio", default=None, help="独立音频文件 (默认从视频提取)")
    parser.add_argument("--size", type=int, default=256, choices=[96, 256])
    parser.add_argument("--output", default="benchmark_output.mp4", help="输出视频路径")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数 (0=全部)")
    parser.add_argument("--fp32", action="store_true", help="强制FP32 (默认自动)")
    parser.add_argument("--no-align", action="store_true", help="禁用面部对齐")
    args = parser.parse_args()

    # --- 加载视频 ---
    cap = cv2.VideoCapture(args.video)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps_video <= 0:
        fps_video = 25.0
    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    print(f"视频: {os.path.basename(args.video)} {total_frames}帧 @{fps_video:.1f}fps {vw}x{vh}")

    # --- 加载音频 ---
    if args.audio and os.path.exists(args.audio):
        audio_full = load_audio(args.audio)
        print(f"音频: {os.path.basename(args.audio)} {len(audio_full)/16000:.1f}s")
    else:
        import tempfile, subprocess
        audio_path = os.path.join(tempfile.gettempdir(), "_bench_audio.wav")
        subprocess.run(
            f'ffmpeg -y -i "{args.video}" -ac 1 -ar 16000 -t {min(total_frames/fps_video + 1, 120)} '
            f'"{audio_path}"', shell=True, capture_output=True, timeout=30)
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            audio_full = load_audio(audio_path)
            os.remove(audio_path)
        else:
            print("警告: 视频无音频流，使用静音")
            audio_full = np.zeros(int(total_frames / fps_video * 16000) + 16000, dtype=np.float32)
        print(f"音频: 从视频提取 {len(audio_full)/16000:.1f}s")

    # --- 加载管线 ---
    print(f"\n加载 Wav2Lip {args.size}x{args.size} 管线...")
    use_align = False if args.no_align else None
    pipeline = StreamingPipeline(
        detect_interval=4, test_audio=False, size=args.size,
        use_align=use_align, use_fp16=not args.fp32)
    pipeline.start()
    print("管线就绪\n")

    # --- 输出视频 (优先H.264兼容编码) ---
    for codec in ['avc1', 'h264', 'mp4v', 'x264', 'XVID']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_ext = 'mp4' if codec in ('avc1', 'h264', 'mp4v', 'x264') else 'avi'
        output_path = args.output if args.output.endswith('.' + out_ext) else f"{os.path.splitext(args.output)[0]}.{out_ext}"
        out = cv2.VideoWriter(output_path, fourcc, fps_video, (vw, h))
        if out.isOpened():
            print(f"输出: {os.path.basename(output_path)} (编码: {codec})")
            break
    if not out.isOpened():
        print("警告: 无法创建输出视频, 请安装 ffmpeg")
        out = None

    # --- 逐帧处理 ---
    latencies = []
    mouth_deltas = []
    detect_times, infer_times, mel_times = [], [], []
    audio_sample_rate = 16000
    audio_pos = 0
    frame_idx = 0

    print(f"开始处理 {max_frames} 帧...")
    t_start = time.perf_counter()

    for i in range(max_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx = i + 1

        t0 = time.perf_counter()

        # 喂音频 (与帧号同步)
        target_audio_pos = int(i / fps_video * audio_sample_rate)
        if target_audio_pos > audio_pos:
            chunk_end = min(target_audio_pos, len(audio_full))
            chunk = audio_full[audio_pos:chunk_end]
            if len(chunk) > 0:
                pipeline.feed_audio(chunk.astype(np.float32))
            audio_pos = chunk_end

        # 推理
        rendered, coords = pipeline.process_frame(frame_bgr)
        t_total = (time.perf_counter() - t0) * 1000
        latencies.append(t_total)

        if rendered is not None:
            frame_bgr, md = composite_face(
                frame_bgr, rendered, *coords, size=args.size)
            mouth_deltas.append(md)
        else:
            mouth_deltas.append(0)

        # 写入输出视频
        if out is not None:
            out.write(frame_bgr)

        # 实时进度
        if frame_idx % 30 == 0 or frame_idx == 1:
            fps = 1000 / np.mean(latencies[-30:]) if len(latencies) >= 30 else 1000 / np.mean(latencies)
            md_avg = np.mean(mouth_deltas[-30:]) if mouth_deltas else 0
            print(f"\r#{frame_idx}/{max_frames}  "
                  f"FPS:{fps:.1f}  Lat:{t_total:.0f}ms  MouthΔ:{md_avg:.1f}/255", end="")

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print("\n")

    # --- 收尾 ---
    if out is not None:
        out.release()
    cap.release()
    pipeline.stop()

    # --- 报告 ---
    lats = np.array(latencies)
    mds = np.array(mouth_deltas)
    face_rate = (np.array(mouth_deltas) > 0.01).mean() * 100

    print("=" * 60)
    print(f"  压测完成 — {args.size}×{args.size} {'FP32' if args.fp32 else 'FP16(自动)'}")
    print("=" * 60)
    print(f"  总帧数:       {len(lats)}")
    print(f"  总耗时:       {elapsed:.1f}s")
    print(f"  平均FPS:      {len(lats)/elapsed:.1f}")
    print(f"  平均延迟:     {lats.mean():.0f}ms")
    print(f"  最小/最大:    {lats.min():.0f}ms / {lats.max():.0f}ms")
    print(f"  P99延迟:      {np.percentile(lats, 99):.0f}ms")
    print(f"  人脸检测率:   {face_rate:.0f}%")
    if len(mds) > 0:
        speaking_frames = mds > 2.0
        if speaking_frames.any():
            print(f"  说话帧嘴Δ:    {mds[speaking_frames].mean():.1f}/255  ({speaking_frames.sum()}帧)")
            silent = ~speaking_frames
            if silent.any():
                print(f"  静音帧嘴Δ:    {mds[silent].mean():.1f}/255  ({silent.sum()}帧)")
        print(f"  平均嘴Δ:      {mds.mean():.1f}/255")
    if out is not None:
        print(f"  输出视频:     {output_path}")
    print("=" * 60)

    # GPU信息
    try:
        import torch
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  GPU:          {torch.cuda.get_device_name(0)}")
            print(f"  Compute Cap:  {cc[0]}.{cc[1]}")
            print(f"  VRAM:         {mem:.1f}GB")
    except:
        pass
    print("=" * 60)


if __name__ == "__main__":
    main()
