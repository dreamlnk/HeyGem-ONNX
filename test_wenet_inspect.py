"""
Inspect WeNet .so module API
Run: wsl bash -c "cd /mnt/d/HeyGem\ ONNX/HeyGem-Linux-Python-Hack-RTX-50 && conda run -n py39 python -u test_wenet_inspect.py"
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wenet'))

print("=== WeNet .so Module Inspection ===\n")

# 1. Try importing compute_ctc_att_bnf
print("1. Trying to import compute_ctc_att_bnf...")
try:
    import compute_ctc_att_bnf
    print(f"   SUCCESS: {compute_ctc_att_bnf}")
    print(f"   File: {compute_ctc_att_bnf.__file__}")

    # List all public API
    attrs = [a for a in dir(compute_ctc_att_bnf) if not a.startswith('_')]
    print(f"   Public API: {attrs}")

    # Try to get function signatures
    for attr in attrs:
        obj = getattr(compute_ctc_att_bnf, attr)
        if callable(obj):
            try:
                import inspect
                sig = inspect.signature(obj)
                print(f"   {attr}{sig}")
            except Exception:
                print(f"   {attr}(...) - no signature available")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

# 2. Try importing other wenet modules
print("\n2. Other wenet modules...")
for mod_name in ['encoder', 'decoder', 'ctc', 'attention', 'embedding', 'subsampling']:
    try:
        mod = __import__(f'transformer.{mod_name}', fromlist=['transformer'])
        print(f"   transformer.{mod_name}: OK")
    except Exception as e:
        print(f"   transformer.{mod_name}: {e}")

# 3. Check wenetmodel.pt
print("\n3. WeNet model checkpoint...")
ckpt_path = 'wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt'
if os.path.exists(ckpt_path):
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        print(f"   Keys: {list(ckpt.keys())[:20]}")
        for k, v in ckpt.items():
            if hasattr(v, 'shape'):
                print(f"   {k}: shape={v.shape}")
            elif isinstance(v, dict):
                print(f"   {k}: dict with {len(v)} entries")
                # Show first few keys
                sub_keys = list(v.keys())[:5]
                for sk in sub_keys:
                    sv = v[sk]
                    if hasattr(sv, 'shape'):
                        print(f"      {sk}: shape={sv.shape}")
    else:
        print(f"   Type: {type(ckpt)}")
else:
    print(f"   NOT FOUND at {ckpt_path}")
    # Search
    import glob
    matches = glob.glob('**/*.pt', recursive=True)
    print(f"   Available .pt files: {matches[:10]}")

# 4. Try to inspect the .so's docstring
print("\n4. Module docstring...")
try:
    import compute_ctc_att_bnf
    if compute_ctc_att_bnf.__doc__:
        print(compute_ctc_att_bnf.__doc__[:1000])
    else:
        print("   No docstring")
except Exception:
    pass

print("\n=== Done ===")
