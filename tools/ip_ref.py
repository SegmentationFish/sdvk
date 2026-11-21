# SPDX-License-Identifier: GPL-3.0-only
# Reference CLIP ViT-H/14 image tower + IP-Adapter image projection in torch (CPU).
# Cross-checks sdvk's --ip-check dumps:
#   OFC_IP_PIX_DUMP  [3,224,224] f32, already rescaled/normalized by sdvk
#   OFC_IP_EMB_DUMP  [1024] f32, projected CLS embedding
#   OFC_IP_TOK_DUMP  [2][4][2048] f32, cond tokens then zero-image tokens
#   OFC_IP_STAGE_DUMP <prefix>_tok/_l0/_l31.f32, patch-embed / block0 / block31 hidden states
# usage: python tools/ip_ref.py <encoder.safetensors> <adapter.safetensors> <dump_dir>
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ENC, AD, DUMP = sys.argv[1], sys.argv[2], sys.argv[3]
STAGE = os.environ.get("REF_STAGE", "ipstage")
SUFFIX = os.environ.get("REF_SUFFIX", "")
EPS = 1e-5
H, HEADS, LAYERS, D, FFN = 16, 16, 32, 1280, 5120


def load(prefix):
    return {k: v.float() for k, v in load_file(prefix).items()}


def rd(name):
    # stage dumps carry the OFC_IP_STAGE_DUMP prefix; prefer it over unrelated *_tok dumps
    stem, ext = os.path.splitext(name)
    for cand in (stem + SUFFIX + ext, STAGE + stem + SUFFIX + ext):
        p = os.path.join(DUMP, cand)
        if os.path.exists(p):
            return np.fromfile(p, dtype="<f4")
    return None


def cmp(tag, ref, mine):
    if mine is None:
        print(f"{tag:12s} missing")
        return
    a = ref.detach().cpu().numpy().ravel().astype(np.float64)
    b = mine.astype(np.float64)
    if a.size != b.size:
        print(f"{tag:12s} size {b.size} != ref {a.size}")
        return
    rel = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    mx = np.abs(a - b).max()
    print(f"{tag:12s} rel {rel:.3e}  cos {cos:.7f}  maxabs {mx:.3e}")


sd = load(ENC)
w = lambda k: sd[k]
b = lambda k: sd[k]


def lin(x, prefix):
    y = x @ w(prefix + ".weight").T
    return y + b(prefix + ".bias") if prefix + ".bias" in sd else y


def ln(x, prefix, d=None):
    d = sd if d is None else d
    m = x.mean(-1, keepdim=True)
    v = x.var(-1, unbiased=False, keepdim=True)
    return (x - m) / torch.sqrt(v + EPS) * d[prefix + ".weight"] + d[prefix + ".bias"]


pix = rd("ip_pixels.f32")
if pix is None:
    sys.exit("ip_pixels.f32 not found in " + DUMP)
x = torch.from_numpy(pix).reshape(3, 224, 224)[None]

# patch embedding: conv 3->D, k=14, s=14 (OpenCLIP has no conv bias; h94 stores one = 0)
pw = w("vision_model.embeddings.patch_embedding.weight")
pb = sd.get("vision_model.embeddings.patch_embedding.bias")
patch = F.conv2d(x, pw, pb, stride=14).flatten(2).transpose(1, 2)  # [1,256,D]
pos = w("vision_model.embeddings.position_embedding.weight")
cls = w("vision_model.embeddings.class_embedding").reshape(1, 1, D)
tok = torch.cat([cls, patch], 1) + pos
cmp("tok", tok, rd("_tok.f32"))

x = ln(tok, "vision_model.pre_layrnorm")
scale = (D // HEADS) ** -0.5
for l in range(LAYERS):
    lp = f"vision_model.encoder.layers.{l}."
    n = ln(x, lp + "layer_norm1")
    q = lin(n, lp + "self_attn.q_proj")
    k = lin(n, lp + "self_attn.k_proj")
    v = lin(n, lp + "self_attn.v_proj")

    def heads(t):
        return t.reshape(1, t.shape[1], HEADS, D // HEADS).transpose(1, 2)

    s = (heads(q) @ heads(k).transpose(-1, -2)) * scale
    p = torch.softmax(s, -1)
    o = (p @ heads(v)).transpose(1, 2).reshape(1, q.shape[1], D)
    o = lin(o, lp + "self_attn.out_proj")
    x1 = x + o
    n = ln(x1, lp + "layer_norm2")
    n = lin(n, lp + "mlp.fc1")
    n = F.gelu(n)
    n = lin(n, lp + "mlp.fc2")
    x = x1 + n
    if l == 0:
        cmp("block0", x, rd("_l0.f32"))
    if l == LAYERS - 1:
        cmp("block31", x, rd("_l31.f32"))

pooled = ln(x[:, :1], "vision_model.post_layernorm")
emb = lin(pooled, "visual_projection")
cmp("embedding", emb.reshape(-1), rd("ip_emb.f32"))

# ---- image projection: Linear(1024 -> 4*2048), reshape, LayerNorm(2048)
ad = load(AD)
e = emb.reshape(1, -1)
t = e @ ad["image_proj.proj.weight"].T + ad["image_proj.proj.bias"]
t = t.reshape(1, 4, 2048)
t = ln(t, "image_proj.norm", ad)
mine = rd("ip_tok.f32")
if mine is not None:
    cmp("tokens cond", t.reshape(-1), mine[: 4 * 2048])
    z = torch.zeros_like(e)
    zt = (z @ ad["image_proj.proj.weight"].T + ad["image_proj.proj.bias"]).reshape(1, 4, 2048)
    zt = ln(zt, "image_proj.norm", ad)
    cmp("tokens zero", zt.reshape(-1), mine[4 * 2048 :])
print("ref ip ok")
