# SPDX-License-Identifier: GPL-3.0-only
# Reference VAE decode in torch (CPU) to cross-check sdvk's Vulkan decoder.
# usage: python tools/refdecode.py <latent.bin> <vae.safetensors> <out.png>
import struct, sys, os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from PIL import Image

_dd = os.environ.get("REF_DUMP_DIR")
def dump(tag, t):
    if _dd:
        os.makedirs(_dd, exist_ok=True)
        np.save(os.path.join(_dd, "ref_" + tag + ".npy"), t.detach().cpu().numpy())

lat_path, vae_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
b = open(lat_path, "rb").read()
d = struct.unpack("<3I", b[4:16])
lat = torch.from_numpy(np.frombuffer(b[16:], dtype="<f4").copy().reshape(1, d[0], d[1], d[2])) / 0.13025
sd = load_file(vae_path)
sd = {k: v.float() for k, v in sd.items()}
W = lambda k: sd[k + ".weight"]
B = lambda k: sd[k + ".bias"]

def conv(x, k, s=1):
    return F.conv2d(x, W(k), B(k), stride=s, padding=1)

def gn(x, k, C):
    h4 = x.reshape(1, 32, C // 32 * x.shape[2] * x.shape[3], 1)
    m = h4.mean(2, keepdim=True); v = h4.var(2, keepdim=True)
    return ((h4 - m) / torch.sqrt(v + 1e-6)).reshape(x.shape) * W(k).reshape(1, C, 1, 1) + B(k).reshape(1, C, 1, 1)

_subcalls = 0
def resnet(x, k, nin):
    global _subcalls
    _subcalls += 1
    sub = os.environ.get("REF_DUMP_SUB") and _subcalls == 1
    h = gn(x, k + ".norm1", x.shape[1])
    h = F.silu(h)  # SDXL VAE act_fn is SiLU, not GELU
    if sub: dump("sub_gn1", h)
    h = conv(h, k + ".conv1")
    if sub: dump("sub_c1", h)
    h = gn(h, k + ".norm2", h.shape[1])
    h = F.silu(h)
    if sub: dump("sub_gn2", h)
    h = conv(h, k + ".conv2")
    if sub: dump("sub_c2", h)
    sc = F.conv2d(x, W(k + ".conv_shortcut"), B(k + ".conv_shortcut")) if nin else x
    return sc + h

def attn(x, k):
    C, H, Wd = x.shape[1], x.shape[2], x.shape[3]
    h4 = x.reshape(1, 32, C // 32 * H * Wd, 1)
    m = h4.mean(2, keepdim=True); v = h4.var(2, keepdim=True)
    h = ((h4 - m) / torch.sqrt(v + 1e-6)).reshape(x.shape) * W(k + ".group_norm").reshape(1, C, 1, 1) + B(
        k + ".group_norm").reshape(1, C, 1, 1)
    def proj(x2, name):
        y = W(name).reshape(C, C) @ x2
        key = name + ".bias"
        return y + sd[key].reshape(-1, 1) if key in sd else y
    q = proj(h.reshape(C, H * Wd), k + ".to_q")
    kk = proj(h.reshape(C, H * Wd), k + ".to_k")
    v = proj(h.reshape(C, H * Wd), k + ".to_v")
    s = (q.T @ kk) / np.sqrt(np.float32(C))
    s = s - s.max(-1, keepdims=True).values
    p = torch.softmax(s, -1)
    o = (v @ p.T).reshape(1, C, H, Wd)
    o = F.conv2d(o, W(k + ".to_out.0").reshape(C, C, 1, 1), B(k + ".to_out.0"))
    return x + o

def up(x, k):
    C = x.shape[1]
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    return F.conv2d(x, W(k), B(k), padding=1)

x = F.conv2d(lat, W("post_quant_conv"), B("post_quant_conv")); dump("postq", x)
x = conv(x, "decoder.conv_in"); dump("conv_in", x)
x = resnet(x, "decoder.mid_block.resnets.0", 0); dump("mid_res0", x)
x = attn(x, "decoder.mid_block.attentions.0"); dump("mid_attn", x)
x = resnet(x, "decoder.mid_block.resnets.1", 0); dump("mid_res1", x)
for blk, (ci, co) in enumerate(((512, 512), (512, 512), (512, 256), (256, 128))):
    for i in range(3):
        x = resnet(x, f"decoder.up_blocks.{blk}.resnets.{i}", co != ci and i == 0); dump(f"up{blk}.res{i}", x)
    if blk < 3:
        x = up(x, f"decoder.up_blocks.{blk}.upsamplers.0.conv"); dump(f"up{blk}.up", x)
h4 = x.reshape(1, 32, 128 // 32 * x.shape[2] * x.shape[3], 1)
m = h4.mean(2, keepdim=True); v = h4.var(2, keepdim=True)
x = ((h4 - m) / torch.sqrt(v + 1e-6)).reshape(x.shape) * W("decoder.conv_norm_out").reshape(1, 128, 1, 1) + B(
    "decoder.conv_norm_out").reshape(1, 128, 1, 1)
x = F.silu(x)  # conv_act is SiLU, not GELU
dump("norm_out", x)
x = conv(x, "decoder.conv_out")
dump("conv_out", x)
img = ((x.clamp(-1, 1) + 1) * 127.5)[0].permute(1, 2, 0).byte().numpy()
Image.fromarray(img).save(out_path)
print("ref decode ok", img.shape)
