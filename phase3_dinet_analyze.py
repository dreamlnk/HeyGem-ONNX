"""
Phase 3: DINet 模型分析 - 从 .pth 权重逆向架构
"""
import numpy
import torch
import torch.serialization
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])

CKPT_PATH = "landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth"

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
face_G = ckpt["face_G"]

print("=== DINet face_G architecture ===")
print(f"Total params: {sum(v.numel() for v in face_G.values()):,}")
print(f"Input size: {ckpt.get('model_input_size')}")
print(f"Output size: {ckpt.get('model_output_size')}")
print(f"Base filters (ngf): {ckpt.get('model_ngf')}")
print(f"Model name: {ckpt.get('model_name')}")
print()

# Group by module path
modules = {}
for k in face_G.keys():
    parts = k.split(".")
    prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
    if prefix not in modules:
        modules[prefix] = []
    modules[prefix].append(k)

print("Modules:")
for mod, keys in sorted(modules.items()):
    shapes = [tuple(face_G[k].shape) for k in keys[:3]]
    print(f"  {mod}: {len(keys)} params, shapes like {shapes}")

# Identify key layers
print("\n--- Key layer shapes ---")
for k, v in sorted(face_G.items()):
    if "conv" in k and "weight" in k and v.dim() >= 2:
        out_ch, in_ch, *spatial = v.shape
        print(f"  {k}: out={out_ch}, in={in_ch}, kernel={spatial}")
    elif "weight" in k and v.dim() == 1:
        print(f"  {k}: dim={v.shape[0]}")
