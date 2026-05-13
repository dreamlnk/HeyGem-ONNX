"""Wav2Lip test with real audio from test video."""
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

# Extract real audio from video
import subprocess
wav_path = '/tmp/test_audio_16k.wav'
subprocess.run([
    'ffmpeg', '-y', '-i',
    '/mnt/d/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50/1004-r.mp4',
    '-ac', '1', '-ar', '16000', '-t', '5',
    wav_path
], capture_output=True)

import librosa
real_audio, sr = librosa.load(wav_path, sr=16000)
print(f'Real audio: {len(real_audio)} samples at {sr}Hz, range=[{real_audio.min():.3f}, {real_audio.max():.3f}]')

# Generate silence for comparison
silence = np.zeros_like(real_audio)

# Generate synthetic speech-like signal (more realistic than pure sine)
tt = np.linspace(0, len(real_audio)/sr, len(real_audio), endpoint=False, dtype=np.float32)
# Mix of harmonics + noise like speech
synth = (np.sin(2*np.pi*150*tt) * 0.3 +
         np.sin(2*np.pi*300*tt) * 0.2 +
         np.sin(2*np.pi*600*tt) * 0.15 +
         np.sin(2*np.pi*1200*tt) * 0.1 +
         np.sin(2*np.pi*2400*tt) * 0.05)
# Amplitude modulation like speech envelope
env = 0.5 + 0.5 * np.sin(2*np.pi*2*tt) * np.sin(2*np.pi*0.5*tt)
synth = synth * env
synth = synth.astype(np.float32) * 0.8

# Load engine
engine = Wav2LipInferenceEngine()
engine.warmup(n=2)

# Prepare face input
face_stack, _ = preprocess_face_wav2lip(face_96)

# Test with different audio sources
audios = [
    ('real_speech', real_audio.astype(np.float32)),
    ('silence', silence),
    ('synth_speech', synth),
]

outputs = {}
print('\n=== Testing different audio sources ===')
for name, audio in audios:
    mel = mel_spectrogram(audio)
    mel_input = get_wav2lip_mel_input(mel)
    mel_t = torch.from_numpy(mel_input)

    with torch.no_grad():
        out = engine.infer(face_stack.unsqueeze(0), mel_t)
    outputs[name] = out
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    print(f'{name:15s}: gray mean={gray.mean():.0f}')

# Compare pairwise
print('\n=== Pairwise comparisons (96x96) ===')
for i, (n1, o1) in enumerate(outputs.items()):
    for j, (n2, o2) in enumerate(outputs.items()):
        if j <= i:
            continue
        diff = np.abs(o1.astype(float) - o2.astype(float))
        full = diff.mean()
        mouth = diff[48:, :, :].mean()
        tight = diff[60:90, 20:76, :].mean()

        # More precise: lips only
        lips = diff[68:85, 25:71, :].mean()
        print(f'{n1:15s} vs {n2:15s}: full={full:.2f} mouth={mouth:.2f} tight={tight:.2f} lips={lips:.2f}')

        if tight > 3.0:
            print(f'  *** STRONG visible mouth movement ***')
        elif tight > 2.0:
            print(f'  ** VISIBLE mouth movement **')
        elif tight > 1.0:
            print(f'  * Subtle movement *')

# Save outputs
for name, out in outputs.items():
    cv2.imwrite(f'/tmp/wl_real_{name}.png', out)

# Per-row comparison: real speech vs silence
print('\n=== Per-row (real speech vs silence, rows 55-95) ===')
real = outputs['real_speech']
sil = outputs['silence']
diff = np.abs(real.astype(float) - sil.astype(float))
for y in range(55, 96):
    row_diff = diff[y, :, :].mean(axis=(0, 1))
    bar = '#' * max(0, int(row_diff))
    print(f'  y={y:2d}: {row_diff:5.2f} {bar}')
