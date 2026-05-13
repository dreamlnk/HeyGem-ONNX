"""Direct Wav2Lip 96x96 output comparison: sweep vs silence."""
import sys, os, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(__file__))
from phase4_wav2lip import Wav2LipInferenceEngine, preprocess_face_wav2lip
from phase2_audio_wav2lip import mel_spectrogram, get_wav2lip_mel_input
from phase1_scrfd_test import load_session, detect

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

# Detect face
detector = load_session()
bboxes, _, _ = detect(detector, img)

if len(bboxes) == 0:
    print("No face detected!")
    sys.exit(1)

bbox = bboxes[0]
x1, y1, x2, y2 = bbox.astype(int)
H, W = img.shape[:2]

# Crop with padding
bw, bh = x2 - x1, y2 - y1
pad_top = int(bh * 0.05)
pad_bottom = int(bh * 0.15)
pad_left = int(bw * 0.05)
pad_right = int(bw * 0.05)

cx1 = max(0, x1 - pad_left)
cy1 = max(0, y1 - pad_top)
cx2 = min(W, x2 + pad_right)
cy2 = min(H, y2 + pad_bottom)

face_crop = img[cy1:cy2, cx1:cx2].copy()
face_96 = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_AREA)

# Generate audio
sr = 16000
dur = 3.0
samples = int(sr * dur)
tt = np.linspace(0, dur, samples, endpoint=False, dtype=np.float32)
freq = 100.0 + 700.0 * (tt / dur)
sweep = np.sin(2.0 * np.pi * freq * tt).astype(np.float32) * 0.8
silence = np.zeros(samples, dtype=np.float32)
sine440 = np.sin(2.0 * np.pi * 440 * tt).astype(np.float32) * 0.8

# Load engine
engine = Wav2LipInferenceEngine()
engine.warmup(n=2)

# Prepare face input (same for all tests)
face_stack, _ = preprocess_face_wav2lip(face_96)

# Test with different audio
audios = [
    ('sweep', sweep),
    ('silence', silence),
    ('sine440', sine440),
]

outputs = {}
for name, audio in audios:
    mel = mel_spectrogram(audio)
    mel_input = get_wav2lip_mel_input(mel)
    mel_t = torch.from_numpy(mel_input)

    with torch.no_grad():
        out = engine.infer(face_stack.unsqueeze(0), mel_t)
    outputs[name] = out
    cv2.imwrite(f'/tmp/wl_{name}.png', out)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    print(f'{name}: gray mean={gray.mean():.0f} min={gray.min()} max={gray.max()}')

# Compare pairwise
print('\n=== Pairwise comparisons (96x96) ===')
for i, (n1, o1) in enumerate(outputs.items()):
    for j, (n2, o2) in enumerate(outputs.items()):
        if j <= i:
            continue
        diff = np.abs(o1.astype(float) - o2.astype(float))
        # Full 96x96
        full = diff.mean()
        # Lower half (rows 48-95): mouth region
        mouth = diff[48:, :, :].mean()
        # Tighter mouth: rows 60-90
        tight = diff[60:90, 20:76, :].mean()
        # Per-row analysis
        row_diffs = diff[48:, :, :].mean(axis=(1, 2))
        max_row_idx = np.argmax(row_diffs) + 48
        print(f'{n1} vs {n2}: full={full:.2f} mouth={mouth:.2f} tight={tight:.2f} '
              f'max_row={max_row_idx}({row_diffs.max():.2f})')

        if tight > 2.0:
            print(f'  *** VISIBLE mouth difference ***')

# Detailed row analysis for sweep vs silence
print('\n=== Per-row analysis (sweep vs silence, rows 40-95) ===')
sweep_out = outputs['sweep']
sil_out = outputs['silence']
diff = np.abs(sweep_out.astype(float) - sil_out.astype(float))
for y in range(40, 96):
    row_diff = diff[y, :, :].mean(axis=(0, 1))
    bar = '#' * max(0, int(row_diff * 2))
    print(f'  y={y:2d}: {row_diff:5.2f} {bar}')
