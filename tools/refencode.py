# SPDX-License-Identifier: GPL-3.0-only
# Reference VAE encode in torch (CPU) to cross-check sdvk's Vulkan encoder.
# usage: python tools/refencode.py <init.png> <vae.safetensors> <out.bin>
# out.bin holds the 4-channel latent in sdvk's file format: "LAT0" + 3x u32 dims + f32 data,
# already multiplied by the scaling factor, i.e. the sampler-space latent.
import sys, os, struct
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from PIL import Image

DUMP = os.environ.get("REF_DUMP_DIR")
FACTOR = float(os.environ.get("REF_FACTOR", "0.13025"))
_subcalls = 0


def dump(tag, t):
    if not DUMP:
        return
    os.makedirs(DUMP, exist_ok=True)
    np.save(os.path.join(DUMP, "ref_" + tag + ".npy"), t.detach().cpu().numpy())


img_path, vae_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sd = {k: v.float() for k, v in load_file(vae_path).items()}
W = lambda k: sd[k + ".weight"]
B = lambda k: sd[k + ".bias"]

im = Image.open(img_path).convert("RGB")
w, h = im.size
# sdvk resamples to the canvas and maps [0,255] to [-1,1]
x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1)[None] / 127.5 - 1.0
print("input %dx%d" % (w, h))


def conv(x, k, s=1, p=1):
    return F.conv2d(x, W(k), B(k), stride=s, padding=p)


def gn(x, k):
    C = x.shape[1]
    h4 = x.reshape(1, 32, C // 32 * x.shape[2] * x.shape[3], 1)
    m = h4.mean(2, keepdim=True)
    v = h4.var(2, keepdim=True)
    y = ((h4 - m) / torch.sqrt(v + 1e-6)).reshape(x.shape)
    return y * W(k).reshape(1, C, 1, 1) + B(k).reshape(1, C, 1, 1)


def resnet(x, k, shortcut):
    global _subcalls
    _subcalls += 1
    h = F.silu(gn(x, k + ".norm1"))
    h = conv(h, k + ".conv1")
    h = F.silu(gn(h, k + ".norm2"))
    h = conv(h, k + ".conv2")
    sc = F.conv2d(x, W(k + ".conv_shortcut"), B(k + ".conv_shortcut")) if shortcut else x
    return sc + h


def attn(x, k):
    C, H, Wd = x.shape[1], x.shape[2], x.shape[3]
    h = F.silu(gn(x, k + ".group_norm"))
    P = h.reshape(C, H * Wd)

    def proj(name):
        return W(name).reshape(C, C) @ P + B(name).reshape(-1, 1)

    q, kk, v = proj(k + ".to_q"), proj(k + ".to_k"), proj(k + ".to_v")
    s = (q.T @ kk) / np.sqrt(np.float32(C))
    s = s - s.max(-1, keepdims=True).values
    p = torch.softmax(s, -1)
    o = (v @ p.T).reshape(1, C, H, Wd)
    o = F.conv2d(o, W(k + ".to_out.0").reshape(C, C, 1, 1), B(k + ".to_out.0"))
    return x + o


# diffusers' Downsample2D pads (0,1,0,1) unconditionally and then runs a padding=0 conv, so the
# output row oh reads original rows 2oh..2oh+2 (the last one may be the zero pad). Padding first and
# then using padding=1 shifts every stage by one pixel.
def down(x, k):
    x = F.pad(x, (0, 1, 0, 1))
    return conv(x, k, s=2, p=0)


x = conv(x, "encoder.conv_in")
dump("enc_conv_in", x)
ci = 128
for blk in range(4):
    co = 128 if blk == 0 else (256 if blk == 1 else 512)
    for i in range(2):
        x = resnet(x, f"encoder.down_blocks.{blk}.resnets.{i}", ci != co and i == 0)
        ci = co
        dump(f"enc_down{blk}.res{i}", x)
    if blk < 3:
        x = down(x, f"encoder.down_blocks.{blk}.downsamplers.0.conv")
        dump(f"enc_down{blk}.down", x)
x = resnet(x, "encoder.mid_block.resnets.0", False)
dump("enc_mid_res0", x)
x = attn(x, "encoder.mid_block.attentions.0")
dump("enc_mid_attn", x)
x = resnet(x, "encoder.mid_block.resnets.1", False)
dump("enc_mid_res1", x)
x = F.silu(gn(x, "encoder.conv_norm_out"))
dump("enc_norm_out", x)
x = conv(x, "encoder.conv_out")
dump("enc_conv_out", x)
x = F.conv2d(x, W("quant_conv"), B("quant_conv"))
dump("enc_quant", x)
mean = x[:, :4] * FACTOR
print("latent %dx%d" % (mean.shape[3], mean.shape[2]))
with open(out_path, "wb") as f:
    f.write(b"OFCL")
    f.write(struct.pack("<3I", 4, mean.shape[2], mean.shape[3]))
    f.write(mean[0].contiguous().numpy().astype("<f4").tobytes())
print("ref encode ok", out_path)
