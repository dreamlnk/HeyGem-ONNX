"""
Phase 3: 直接加载 DINet 模型进行逐帧推理
绕过 create_model 工厂，手动构造选项并加载权重
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"


class SimpleOpt:
    """容错选项对象，缺失属性返回 None"""
    def __getattr__(self, name):
        return None


def load_dinet_model():
    """直接加载 DINet 模型 + 权重，返回可推理的 PyTorch 模型"""
    from landmark2face_wy.models.l2faceaudio_model import L2FaceAudioModel

    opt = SimpleOpt()
    opt.model = "l2faceaudio"
    opt.name = "anylang"
    opt.img_size = 256
    opt.audio_feature = "wenet"
    opt.feature_path = "./landmark2face_wy/feature"
    opt.batch_size = 1
    opt.gpu_ids = [0]
    opt.isTrain = False
    opt.checkpoints_dir = "./landmark2face_wy/checkpoints"
    # Model architecture options
    opt.preprocess = "resize_and_colorjitter"
    opt.no_flip = True
    opt.load_iter = 0
    opt.epoch = "latest"
    opt.ngf = 64
    opt.norm = "instance"
    opt.no_dropout = True
    opt.init_type = "normal"
    opt.init_gain = 0.02
    opt.verbose = False
    opt.suffix = ""
    opt.direction = "AtoB"
    opt.phase = "test"
    opt.num_test = 1000
    opt.serial_batches = True
    opt.num_threads = 4
    opt.padding_type = "reflect"
    opt.pool_size = 0
    opt.lr_policy = "linear"
    opt.epoch_count = 1
    opt.n_layers_D = 3
    opt.ndf = 64
    opt.no_ganFeat_loss = True
    opt.no_vgg_loss = True
    opt.lambda_L1 = 100.0
    opt.netG = "DINet"
    opt.netD = "basic"
    opt.input_nc = 3
    opt.output_nc = 3
    opt.display_winsize = 256
    opt.aspect_ratio = 1.0
    opt.crop_size = 256
    opt.max_dataset_size = 10000
    opt.display_freq = 100
    opt.save_epoch_freq = 5
    opt.print_freq = 100
    opt.save_latest_freq = 5000
    opt.continue_train = False
    opt.lr = 0.0002
    opt.lr_decay_iters = 50
    opt.beta1 = 0.5
    opt.use_wandb = False

    # 尝试创建模型
    print("Creating L2FaceAudioModel...")
    model = L2FaceAudioModel(opt)
    print(f"Model type: {type(model).__name__}")

    # 加载 checkpoint
    import numpy
    import torch.serialization
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # 获取生成器并加载权重
    if hasattr(model, 'netG'):
        model.netG.load_state_dict(ckpt["face_G"], strict=False)
        print(f"Loaded {len(ckpt['face_G'])} weights into netG")
    elif hasattr(model, 'netG_A'):
        model.netG_A.load_state_dict(ckpt["face_G"], strict=False)

    model.eval()
    model.to("cuda")
    print(f"Params: {sum(p.numel() for p in model.netG.parameters()):,}")
    return model


def test_inference(model):
    """测试推理: 创建虚拟输入，跑一遍推理"""
    print("\n--- 推理测试 ---")

    # DINet 需要的输入
    dummy_source = torch.randn(1, 3, 256, 256).cuda()      # 源人脸
    dummy_target = torch.randn(1, 3, 256, 256).cuda()      # 目标人脸
    dummy_audio_s = torch.randn(1, 1, 256, 256).cuda()     # 源音频特征
    dummy_audio_t = torch.randn(1, 1, 256, 256).cuda()     # 目标音频特征
    dummy_mask_s = torch.ones(1, 1, 256, 256).cuda()       # 源mask
    dummy_mask_t = torch.ones(1, 1, 256, 256).cuda()       # 目标mask

    netG = model.netG

    # 尝试不同输入组合
    with torch.no_grad():
        # 尝试1: 标准调用 (source, target, audio_s, audio_t, mask_s, mask_t)
        try:
            output = netG(dummy_source, dummy_target, dummy_audio_s, dummy_audio_t, dummy_mask_s, dummy_mask_t)
            print(f"Output shape: {output.shape}")
            ok = True
        except Exception as e1:
            print(f"6-arg call failed: {e1}")
            # 尝试2: 只用3个参数
            try:
                output = netG(dummy_source, dummy_audio_s, dummy_mask_s)
                print(f"3-arg output: {output.shape}")
                ok = True
            except Exception as e2:
                print(f"3-arg call failed: {e2}")
                ok = False

    if not ok:
        print("Could not determine model input format")
        return

    # Warmup
    print("Warmup...")
    torch.cuda.synchronize()
    for _ in range(10):
        netG(dummy_source, dummy_target, dummy_audio_s, dummy_audio_t, dummy_mask_s, dummy_mask_t)
    torch.cuda.synchronize()

    # Benchmark
    print("Benchmark...")
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        netG(dummy_source, dummy_target, dummy_audio_s, dummy_audio_t, dummy_mask_s, dummy_mask_t)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"\nDINet 推理速度:")
    print(f"  平均: {times.mean():.1f}ms  ({1000/times.mean():.0f} FPS)")
    print(f"  中位: {np.median(times):.1f}ms")
    print(f"  最小: {times.min():.1f}ms / 最大: {times.max():.1f}ms")

    # 显存使用
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        mem_max = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n显存: 当前={mem:.1f}GB, 峰值={mem_max:.1f}GB")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: DINet 直接加载 + 推理")
    print("=" * 60)
    model = load_dinet_model()
    test_inference(model)
    print("\nPhase 3 完成 ✓")
