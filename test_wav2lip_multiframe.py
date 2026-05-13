"""Multi-frame Wav2Lip test: verify temporal consistency and visible mouth movement."""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from pipeline_wav2lip import StreamingPipeline
from phase1_scrfd_test import detect

# Use test video
video_path = '/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/1004-r.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open video")
    sys.exit(1)

# Get first frame
ret, img = cap.read()
if not ret:
    print("Cannot read first frame")
    sys.exit(1)

img = cv2.resize(img, (1280, 720))
print(f'Frame: {img.shape}')

# Generate alternating audio (speech/silence)
sr = 16000
dur = 3.0
samples = int(sr * dur)
tt = np.linspace(0, dur, samples, endpoint=False, dtype=np.float32)
freq = 100.0 + 700.0 * (tt / dur)
sweep = np.sin(2.0 * np.pi * freq * tt).astype(np.float32) * 0.8
silence = np.zeros(samples, dtype=np.float32)

# Init pipeline
pipeline = StreamingPipeline(detect_interval=2, test_audio=False)
pipeline.start()

# Process 30 frames alternating sweep/silence every 2 frames
print(f'\nProcessing 30 frames with alternating audio...')
outputs = []
for i in range(30):
    if (i // 2) % 2 == 0:
        pipeline.audio_buffer = sweep.copy()
    else:
        pipeline.audio_buffer = silence.copy()

    out = pipeline.process_frame(img.copy())
    outputs.append(out)
    if out is None:
        print(f'  Frame {i}: NO OUTPUT')
        break

print(f'\n=== Frame-to-frame comparison ===')
diffs = []
for i in range(1, len(outputs)):
    d = np.abs(outputs[i].astype(float) - outputs[i-1].astype(float)).mean()
    diffs.append(d)
    # Check face ROI specifically
    bboxes, _, _ = detect(pipeline.detector, outputs[i])
    if len(bboxes) > 0:
        x1, y1, x2, y2 = bboxes[0].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(outputs[i].shape[1], x2)
        y2 = min(outputs[i].shape[0], y2)
        face_d = np.abs(outputs[i][y1:y2, x1:x2].astype(float) -
                        outputs[i-1][y1:y2, x1:x2].astype(float)).mean()
        mouth_y = y1 + (y2 - y1) * 2 // 3
        mouth_d = np.abs(outputs[i][mouth_y:y2, x1:x2].astype(float) -
                         outputs[i-1][mouth_y:y2, x1:x2].astype(float)).mean()
        audio_label = 'SPEECH' if (i // 2) % 2 == 1 else 'SILENT'
        print(f'  {i-1}->{i} ({audio_label}): full={d:.2f} face={face_d:.2f} mouth={mouth_d:.2f}')

print(f'\nAvg full diff: {np.mean(diffs):.2f}')
print(f'Max full diff: {np.max(diffs):.2f}')

# Save last output for visual inspection
cv2.imwrite('/tmp/wav2lip_multiframe_last.png', outputs[-1])
print(f'Saved /tmp/wav2lip_multiframe_last.png')
