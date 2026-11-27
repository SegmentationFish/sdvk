"""Reference MiniMax Music 3 flow-VAE decoder (torch CPU) for the sdvk port.

Reads the published `vocoder/` safetensors (weight-norm g/v pairs), folds the weight norm and
runs the DAC-style decoder on a raw [frames, 128] f32 latent. Writes the stereo result as raw
f32 and a wav, then the C++ side can be diffed against it.

usage: python tools/mm3_vae_ref.py <dav.safetensors> <latent.f32> <frames> <out_prefix>
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

VAE, LAT, FRAMES, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sd = load_file(VAE)


def wn(prefix):
    if prefix + ".weight_v" not in sd:  # dec_in_proj is a plain conv (no weight norm)
        return sd[prefix + ".weight"].float(), sd[prefix + ".bias"].float()
    v = sd[prefix + ".weight_v"].float()
    g = sd[prefix + ".weight_g"].float()
    n = v.reshape(v.shape[0], -1).norm(dim=1).clamp_min(1e-12)
    w = v * (g.reshape(-1, 1, 1) / n.reshape(-1, 1, 1))
    b = sd.get(prefix + ".bias")
    return w, (b.float() if b is not None else None)


def conv(x, prefix, stride=1, pad=0, dil=1):
    w, b = wn(prefix)
    return F.conv1d(x, w, b, stride=stride, padding=pad, dilation=dil)


def convT(x, prefix, stride, pad):
    w, b = wn(prefix)
    return F.conv_transpose1d(x, w, b, stride=stride, padding=pad)


def snake(x, prefix):
    a = sd[prefix + ".alpha"].float()
    return x + torch.sin(x * a) ** 2 / (a + 1e-9)


lat = np.fromfile(LAT, dtype="<f4").reshape(FRAMES, 128)
lat = torch.from_numpy(lat.copy()).t().unsqueeze(0)  # [1, 128, T]

out = torch.zeros(1, 2, FRAMES * 512)
for side in range(2):
    x = lat[:, side * 64 : (side + 1) * 64]
    x = conv(x, "dec_in_proj")           # 64 -> 1024, k=1
    x = conv(x, "decoder.model.0", pad=3)  # 1024 -> 1536, k=7
    strides = [8, 8, 4, 2]
    for b in range(4):
        bp = f"decoder.model.{b+1}."
        s = strides[b]
        x = snake(x, bp + "block.0")
        x = convT(x, bp + "block.1", stride=s, pad=s // 2)
        for r in range(3):
            rp = bp + f"block.{r+2}."
            y = snake(x, rp + "block.0")
            y = conv(y, rp + "block.1", pad=3 * [1, 3, 9][r], dil=[1, 3, 9][r])
            y = snake(y, rp + "block.2")
            y = conv(y, rp + "block.3")
            x = x + y
    x = snake(x, "decoder.model.5")
    x = conv(x, "decoder.model.6", pad=3)  # 96 -> 1
    x = torch.tanh(x)
    n = min(x.shape[-1], FRAMES * 512)
    out[0, side, :n] = x[0, 0, :n]

arr = out[0].numpy()
arr.tofile(OUT + "_ref.f32")
print("ref:", arr.shape, "peak %.4f rms %.4f" % (np.abs(arr).max(), np.sqrt((arr ** 2).mean())))
with open(OUT + "_ref.wav", "wb") as f:
    pcm = (np.clip(arr.T, -1, 1) * 32767).astype("<i2")
    f.write(b"RIFF" + (36 + pcm.nbytes).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little"))
    f.write((1).to_bytes(2, "little") + (2).to_bytes(2, "little") + (44100).to_bytes(4, "little"))
    f.write((44100 * 4).to_bytes(4, "little") + (4).to_bytes(2, "little") + (16).to_bytes(2, "little"))
    f.write(b"data" + pcm.nbytes.to_bytes(4, "little") + pcm.tobytes())
