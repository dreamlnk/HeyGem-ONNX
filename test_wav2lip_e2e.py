"""Wav2Lip E2E test: sweep vs silence mouth movement comparison."""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from pipeline_wav2lip import StreamingPipeline
from phase1_scrfd_test import detect

# Get test image
img = cv2.imread('/tmp/test_frame.png')
if img is None:
    cap = cv2.VideoCapture('/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/1004-r.mp4')
    cap.read()
    ret, img = cap.read()
    cap.release()
    if img is None:
        print("NO TEST IMAGE")
        sys.exit(1)
    img = cv2.resize(img, (1280, 720))

print(f'Test frame: {img.shape}')

# Init pipeline
pipeline = StreamingPipeline(detect_interval=1, test_audio=False)
pipeline.start()

# Generate test audio
sr = 16000
dur = 3.0
samples = int(sr * dur)
tt = np.linspace(0, dur, samples, endpoint=False, dtype=np.float32)
freq = 100.0 + 700.0 * (tt / dur)
sweep = np.sin(2.0 * np.pi * freq * tt).astype(np.float32) * 0.8
silence = np.zeros(samples, dtype=np.float32)

# Test 1: sweep (speech)
print('\n=== Test 1: Sweep ===')
pipeline.audio_buffer = sweep.copy()
pipeline.prev_face_norm = None
out_sweep = pipeline.process_frame(img.copy())
if out_sweep is not None:
    cv2.imwrite('/tmp/wav2lip_out_sweep.png', out_sweep)
    gray = cv2.cvtColor(out_sweep, cv2.COLOR_BGR2GRAY)
    print(f'Output: {out_sweep.shape}, gray mean={gray.mean():.0f}')

# Test 2: silence
print('\n=== Test 2: Silence ===')
pipeline.audio_buffer = silence.copy()
pipeline.prev_face_norm = None
out_silence = pipeline.process_frame(img.copy())
if out_silence is not None:
    cv2.imwrite('/tmp/wav2lip_out_silence.png', out_silence)

# Compare
if out_sweep is not None and out_silence is not None:
    diff = np.abs(out_sweep.astype(float) - out_silence.astype(float))
    print(f'\n=== Comparison ===')
    print(f'Whole frame diff: {diff.mean():.1f}')

    # Face ROI
    bboxes, _, _ = detect(pipeline.detector, out_sweep)
    if len(bboxes) > 0:
        x1, y1, x2, y2 = bboxes[0].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(out_sweep.shape[1], x2)
        y2 = min(out_sweep.shape[0], y2)
        face_diff = diff[y1:y2, x1:x2]
        print(f'Face ROI diff: {face_diff.mean():.1f}')
        # Mouth: lower third
        mouth_y = y1 + (y2 - y1) * 2 // 3
        mouth_diff = diff[mouth_y:y2, x1:x2]
        print(f'Mouth ROI diff: {mouth_diff.mean():.1f}')
        print(f'Mouth max diff: {mouth_diff.max():.0f}')

        threshold = 5.0
        if mouth_diff.mean() > threshold:
            print(f'*** SUCCESS: Mouth movement {mouth_diff.mean():.1f} > {threshold} ***')
        elif mouth_diff.mean() > 2.0:
            print(f'** MARGINAL: {mouth_diff.mean():.1f}, some movement **')
        else:
            print(f'* FAIL: {mouth_diff.mean():.1f} < 2.0, no visible movement *')
    else:
        print('No face detected in output')
