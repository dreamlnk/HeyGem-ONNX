# HeyGem — 实时AI数字人唇同步系统 (Wav2Lip Streaming)

## 架构概览

```
┌─────────────────────┐     TCP(BGR原始流)     ┌──────────────────────┐
│  Windows 客户端       │ ◄──────────────────► │  WSL Linux 服务端     │
│                      │   96×96 面部数据       │                      │
│  · 摄像头/视频输入    │   + 裁剪坐标          │  · YuNet 人脸检测     │
│  · 麦克风/音频输入    │                      │  · Wav2Lip 推理       │
│  · Delta面部合成      │                      │  · 梅尔频谱提取        │
│  · 虚拟摄像头输出      │                      │  · GPU CUDA加速       │
└─────────────────────┘                      └──────────────────────┘
```

## 核心技术

### Wav2Lip 模型 (96×96 / 256×256, FP32)
```
输入: 人脸S×S (6通道=[masked_lower_BGR, full_BGR]) + 梅尔频谱(1,80,16)
  ┌─ Face Encoder: 7-8层conv stride=2, 6ch→512ch, S→1
  ├─ Audio Encoder: 4层conv stride=2, 1ch→512ch, (80,16)→(5,1)
  └─ Decoder:       7-8层转置卷积, 512ch→3ch, 1→S
输出: 唇同步人脸 S×S BGR (S=96 or 256)
```

### 音频处理
- 16kHz → 预加重(0.97) → STFT(n_fft=800, hop=200, win=800)
- 80维梅尔谱 (fmin=55, fmax=7600) → dB → 对称归一化[-4,4]
- 取最后16帧 (200ms上下文) 作为模型输入

### 人脸检测
- YuNet (OpenCV FaceDetectorYN), 5点关键点 + 皮肤颜色验证
- 每2帧检测一次, 5帧bbox移动平均平滑

### Delta合成方案 (核心创新)
```
1. 服务端: Wav2Lip生成96×96渲染人脸 → 发送给客户端
2. 客户端: 本地将原始高清面部缩放至96×96
3. 计算 delta = 渲染人脸 - 原始人脸(96×96)  [无精度损失]
4. 升采样delta至裁剪分辨率
5. 原始高清面部 + delta → 保留100%纹理细节
6. 椭圆形羽化遮罩, 仅作用于口部区域
```
- 优势: 原始面部纹理100%保留, 无模糊, 无可见接缝

### 音画同步
- 帧号驱动: `audio_pos = frame_count / fps × 16000`
- 消除墙上时钟漂移, 帧N永远对应音频位置N/fps×16000
- 音频播放从当前视频帧位置开始, 对齐画面

## 性能指标

| 指标 | 96×96 | 256×256 |
|------|-------|---------|
| 推理延迟 | ~15ms/帧 | ~15-25ms/帧 |
| 口部运动量 | 9.6%/255 (人眼清晰可见, 阈值6%) | 同等或更优 |
| 口部清晰度 | 基准 | ~7×像素提升 |
| 帧率自适应 | 25fps (视频) / 30fps (摄像头) | 25fps (视频) / 30fps (摄像头) |
| 网络带宽 | ~700KB/s (TCP BGR原始流) | ~5.3MB/s |
| 模型权重 | 436MB (wav2lip_gan.pth) | 205MB (wav2lip256.pth) |
| 音频采样率 | 16000Hz | 16000Hz |

## 快速开始

### 1. 启动服务端 (WSL/Linux)
```bash
cd "D:/HeyGem ONNX/HeyGem-Linux-Python-Hack-RTX-50"
# 96×96 模式 (默认)
python stream_server_tcp.py
# 256×256 模式 (更高清晰度)
python stream_server_tcp.py --size 256
```
等待模型加载完成, 看到 "监听端口 7863..." 即可。

### 2. 启动客户端 (Windows)

**注意**: 客户端 `--size` 参数必须与服务端一致。

**摄像头 + 麦克风 (实时数字人)**
```bash
python windows_client_tcp.py --mic [--size 256]
```

**视频文件 + 麦克风 (视频画面+实时口型)**
```bash
python windows_client_tcp.py --video input.mp4 --loop --mic [--size 256]
```

**视频文件 + 独立音频**
```bash
python windows_client_tcp.py --video input.mp4 --audio audio.mp3 --loop [--size 256]
```

**虚拟摄像头模式 (OBS推流)**
```bash
python windows_client_tcp.py --video input.mp4 --virtualcam --loop --mic [--size 256]
```

**全部参数**
```
--video PATH       视频文件路径 (不传则使用摄像头)
--audio PATH       独立音频文件 (不传则使用视频原声或麦克风)
--mic              使用麦克风驱动嘴型
--virtualcam       输出到OBS虚拟摄像头
--loop             视频循环播放
--portrait         竖屏模式 720×1280
--camera ID        摄像头ID (默认0)
```

### 3. 退出
按 `Q` 或 `Ctrl+C`

## 文件结构

```
stream_server_tcp.py          # TCP服务端入口 (WSL)
windows_client_tcp.py          # Windows客户端入口
pipeline_wav2lip.py            # 流式管线 (检测→裁剪→推理)
phase1_scrfd_test.py           # YuNet人脸检测
phase2_audio_wav2lip.py        # 梅尔频谱提取
phase4_wav2lip.py              # Wav2Lip推理引擎 (支持96/256)
models_wav2lip/
  ├─ wav2lip.py                # 96×96 模型定义
  ├─ wav2lip256.py             # 256×256 模型定义
  └─ conv.py                   # Conv2d/Conv2dTranspose 基础模块
pretrain_models/               # 权重文件目录
  ├─ wav2lip_gan.pth          # 96×96 权重 (~436MB)
  └─ wav2lip256.pth           # 256×256 权重 (~205MB)
```

## 协议

TCP二进制协议, 无编解码开销:

**客户端 → 服务端:**
```
[1字节 type][4字节 len][payload]
type=0: 视频帧 (width, height, BGR数据)
type=1: 音频块 (float32样本)
type=2: 重置 (清空音频缓冲)
```

**服务端 → 客户端:**
```
[4字节 total_len][8字节 坐标(int16×4)][27648字节 96×96×3 BGR]
人脸未检测到时: [4字节 0]
```

## 与DINet对比

| 对比维度 | DINet (已废弃) | Wav2Lip |
|----------|---------------|---------|
| 核心目标 | 面部重演 | 唇同步 |
| 嘴部运动量 | 0.6%/255 (不可见) | 9.6%/255 (清晰可见) |
| 差异倍数 | - | **16倍** |
| 音频驱动 | adaAT调制 (弱信号) | 直接编码器 (强信号) |
| 分辨率 | 256×256 | 96×96 + Delta合成 |

## 依赖

- Python 3.11 (Windows) / 3.9 (Linux)
- PyTorch ≥ 2.0 + CUDA
- OpenCV (cv2)
- NumPy, SciPy, librosa
- sounddevice (麦克风/音频播放)
- pyvirtualcam (虚拟摄像头, 可选)
