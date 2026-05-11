"""
Phase 2b: 音频特征提取 (MFCC)
用于实时流式音频特征提取，替代 wenet ASR 批处理方案
"""
import os
import time
import numpy as np
import librosa

SAMPLE_RATE = 16000
FEATURE_DIM = 40       # MFCC 系数
WINDOW_SIZE = 3.0      # 滑动窗口大小(秒)，覆盖最近的3秒音频
HOP_SIZE = 0.5         # 窗口滑动步长(秒)
TARGET_LEN = 256       # DINet 模型期望的音频特征长度


def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=FEATURE_DIM):
    """提取 MFCC + delta + delta-delta 特征

    Args:
        audio: 1D numpy array, 音频采样
        sr: 采样率
        n_mfcc: MFCC 系数个数

    Returns:
        features: [time_frames, n_mfcc*3] 特征矩阵 (MFCC + Δ + ΔΔ)
    """
    if len(audio) < sr * 0.025:  # 至少25ms
        return np.zeros((1, n_mfcc * 3), dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=sr, n_mfcc=n_mfcc)  # [n_mfcc, time]
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)  # [n_mfcc*3, time]
    return features.T  # [time, n_mfcc*3]


def extract_mfcc_streaming(audio_chunk, sr=SAMPLE_RATE, n_mfcc=FEATURE_DIM):
    """流式版本: 单次调用只提取 MFCC，不做 delta (需要上下文)"""
    if len(audio_chunk) < sr * 0.025:
        return np.zeros((1, n_mfcc), dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=audio_chunk.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # [time, n_mfcc]


def prepare_dinet_input(features, target_len=TARGET_LEN):
    """将音频特征调整为 DINet 模型输入格式 [256, 256]"""
    time_frames, feat_dim = features.shape

    # 截断或填充到 target_len
    if time_frames > target_len:
        features = features[:target_len, :]
    else:
        pad = np.zeros((target_len - time_frames, feat_dim), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)

    # 特征维度也填充到 256
    if feat_dim < 256:
        pad = np.zeros((target_len, 256 - feat_dim), dtype=np.float32)
        features = np.concatenate([features, pad], axis=1)
    elif feat_dim > 256:
        features = features[:, :256]

    return features  # [256, 256]


def test_with_file(audio_path):
    """用完整音频文件测试"""
    print(f"\n加载音频: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr
    print(f"  采样率: {sr}Hz, 时长: {duration:.1f}s, 采样点: {len(audio)}")

    # 完整提取
    t0 = time.perf_counter()
    features = extract_mfcc(audio)
    dt_full = (time.perf_counter() - t0) * 1000
    print(f"  完整MFCC提取: {dt_full:.1f}ms, 维度: {features.shape}")
    print(f"  音频时长/特征帧 = {duration/features.shape[0]*1000:.1f}ms per frame")

    # DINet 输入格式
    dinet_inp = prepare_dinet_input(features)
    print(f"  DINet输入: {dinet_inp.shape}")

    # 模拟流式提取 (滑动窗口)
    print(f"\n--- 流式模拟 (窗口={WINDOW_SIZE}s, 步长={HOP_SIZE}s) ---")
    stream_times = []
    num_windows = int((duration - WINDOW_SIZE) / HOP_SIZE) + 1
    for i in range(max(1, num_windows)):
        start_sample = int(i * HOP_SIZE * sr)
        end_sample = int(start_sample + WINDOW_SIZE * sr)
        chunk = audio[start_sample:end_sample]

        t0 = time.perf_counter()
        feat = extract_mfcc(chunk)
        dinet_inp = prepare_dinet_input(feat)
        dt = (time.perf_counter() - t0) * 1000
        stream_times.append(dt)

    if stream_times:
        ts = np.array(stream_times)
        print(f"  窗口数: {len(stream_times)}")
        print(f"  每窗口耗时: avg={ts.mean():.1f}ms, max={ts.max():.1f}ms")
        print(f"  开销比: {ts.mean()/(HOP_SIZE*1000)*100:.1f}% (特征提取/窗口时长)")

    return features


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2b: 音频特征提取验证")
    print("=" * 60)

    audio_path = "/tmp/test_audio.wav"

    # 从示例视频提取音轨
    video_path = "/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/example/video.mp4"
    if os.path.exists(video_path):
        print("从示例视频提取音轨...")
        os.system(f"ffmpeg -y -i '{video_path}' -ac 1 -ar {SAMPLE_RATE} -t 10 {audio_path} 2>/dev/null")

    if os.path.exists(audio_path):
        test_with_file(audio_path)
    else:
        # 生成合成音频测试
        print("生成合成音频进行测试...")
        sr = SAMPLE_RATE
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz 正弦波
        features = extract_mfcc(audio)
        print(f"  合成音频: {duration}s, 特征维度: {features.shape}")
        dinet_inp = prepare_dinet_input(features)
        print(f"  DINet输入: {dinet_inp.shape}")

    print("\nPhase 2b 完成 ✓")
