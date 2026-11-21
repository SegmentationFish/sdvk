# SPDX-License-Identifier: GPL-3.0-only
# Cross-check the IP-Adapter per-layer K/V projection in sdvk against torch.
#   engine dump: OFC_IP_KV_DUMP=<prefix> from `sdvk --ip-check`, plus OFC_IP_TOK_DUMP=<file>
#   usage: python tools/ip_kv_cmp.py <adapter.safetensors> <dump_prefix> [tokens.f32]
# Layout of <prefix>_k.f32 / _v.f32: per layer, 4 cond tokens then 4 zero-image tokens,
# each token `layerDim` wide, layers packed back to back in adapter index order.
import sys
import numpy as np
from safetensors.torch import load_file

AD, PREFIX = sys.argv[1], sys.argv[2]
TOK = sys.argv[3] if len(sys.argv) > 3 else PREFIX[: PREFIX.rfind("/") + 1] + "ip_tok.f32"

meta = open(PREFIX + "_layers.txt").read().split("\n")
L, T, DIM = (int(v) for v in meta[0].split())
layers = []
for line in meta[1:]:
    if not line.strip():
        continue
    idx, dim, off = line.split()
    layers.append((int(idx), int(dim), int(off)))
assert len(layers) == L, (len(layers), L)

toks = np.fromfile(TOK, dtype="<f4")
assert toks.size == 2 * T * DIM, toks.size
cond = toks[: T * DIM].reshape(T, DIM)
zero = toks[T * DIM :].reshape(T, DIM)

K = np.fromfile(PREFIX + "_k.f32", dtype="<f4")
V = np.fromfile(PREFIX + "_v.f32", dtype="<f4")
ad = load_file(AD)


def rel(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30), float(
        a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    )


worst = (0.0, 0.0, None)
for idx, dim, off in layers:
    for br, tok in ((0, cond), (1, zero)):
        base = off + br * T * dim
        exp_k = tok @ ad[f"ip_adapter.{idx}.to_k_ip.weight"].float().numpy().T
        exp_v = tok @ ad[f"ip_adapter.{idx}.to_v_ip.weight"].float().numpy().T
        got_k = K[base : base + T * dim].reshape(T, dim)
        got_v = V[base : base + T * dim].reshape(T, dim)
        rk, ck = rel(exp_k, got_k)
        rv, cv = rel(exp_v, got_v)
        layer_max = max(rk, rv)
        if layer_max > worst[0]:
            worst = (layer_max, min(ck, cv), (idx, dim, br, rk, rv))
        if br == 0 and (rk > 2e-2 or rv > 2e-2):
            print(f"layer {idx:3d} dim {dim:4d} relK {rk:.2e} relV {rv:.2e}  <-- high")
print(f"{L} layers x {T} tokens, dim {DIM}: worst rel {worst[0]:.3e} (cos {worst[1]:.6f}) "
      f"at {worst[2]}")
print("ip kv ok" if worst[0] < 3e-2 else "ip kv MISMATCH")
