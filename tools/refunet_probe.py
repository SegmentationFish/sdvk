# SPDX-License-Identifier: GPL-3.0-only
# Reference SDXL UNet forward in torch (CPU) straight from the Q8_0 GGUF, mirroring
# sdvkgen's unetForward structure 1:1. usage:
#   python tools/refunet_probe.py <unet.gguf> <lat_raw.f32> <ctx.f32> <pooled.f32> <t>
import struct, sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(16)
gguf_path, lat_path, ctx_path, pooled_path, t = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])

f = open(gguf_path, "rb")
magic, ver, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
assert magic == b"GGUF"

def rd_str():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")

def rd_val(vt):
    if vt in (0, 7): return struct.unpack("<B", f.read(1))[0]
    if vt in (1,): return struct.unpack("<b", f.read(1))[0]
    if vt == 2: return struct.unpack("<H", f.read(2))[0]
    if vt == 3: return struct.unpack("<h", f.read(2))[0]
    if vt == 4: return struct.unpack("<I", f.read(4))[0]
    if vt == 5: return struct.unpack("<i", f.read(4))[0]
    if vt == 6: return struct.unpack("<f", f.read(4))[0]
    if vt == 8: return rd_str()
    if vt == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n): rd_val(et)
        return None
    if vt == 10: return struct.unpack("<Q", f.read(8))[0]
    if vt == 11: return struct.unpack("<q", f.read(8))[0]
    if vt == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"vtype {vt}")

for _ in range(n_kv):
    k = rd_str()
    vt = struct.unpack("<I", f.read(4))[0]
    rd_val(vt)

T = {}
for _ in range(n_tensors):
    name = rd_str()
    nd = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)][::-1]
    ttype = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    T[name] = (tuple(dims), ttype, off)
data_base = (f.tell() + 31) & ~31

def get(name):
    shape, ttype, off = T[name]
    ne = int(np.prod(shape))
    f.seek(data_base + off)
    if ttype == 8:
        raw = f.read(ne // 32 * 34)
        nb = len(raw) // 34
        out = np.empty(nb * 32, dtype=np.float32)
        for i in range(nb):
            blk = raw[i * 34:(i + 1) * 34]
            s = np.frombuffer(blk[:2], dtype="<f2").astype(np.float32)[0]
            q = np.frombuffer(blk[2:], dtype=np.int8).astype(np.float32)
            out[i * 32:(i + 1) * 32] = q * s
        arr = out.reshape(shape)
    elif ttype == 0:
        arr = np.frombuffer(f.read(ne * 4), dtype="<f4").copy().reshape(shape)
    elif ttype == 1:
        arr = np.frombuffer(f.read(ne * 2), dtype="<f2").astype(np.float32).reshape(shape)
    else:
        raise ValueError(f"type {ttype}")
    return torch.from_numpy(arr)

cache = {}
def W(k):
    if k not in cache: cache[k] = get(k + ".weight")
    return cache[k]
def B(k):
    key = "b:" + k
    if key not in cache:
        cache[key] = get(k + ".bias") if k + ".bias" in T else None
    return cache[key]

def lin(x, k):
    w = W(k)
    y = x @ w.reshape(w.shape[0], -1).T
    b = B(k)
    return y + b if b is not None else y

def conv1(x, k):
    w = W(k).reshape(W(k).shape[0], -1, 1, 1)
    return F.conv2d(x, w, B(k))

def conv3(x, k, s=1):
    return F.conv2d(x, W(k), B(k), stride=s, padding=1)

def gn(x, k, eps):
    C = x.shape[1]
    h = x.reshape(1, 32, C // 32 * x.shape[2], x.shape[3])
    m = h.mean(2, keepdim=True); v = h.var(2, keepdim=True)
    return ((h - m) / torch.sqrt(v + eps)).reshape(x.shape) * W(k).reshape(1, -1, 1, 1) + B(k).reshape(1, -1, 1, 1)

def attn(x, k, heads, ctx=None):
    Bn, C, H, Wd = x.shape
    P = H * Wd
    h = x.reshape(Bn, C, P).transpose(1, 2)
    q = lin(h, k + ".to_q").reshape(Bn, P, heads, C // heads).transpose(1, 2)
    kk = lin(ctx, k + ".to_k").reshape(Bn, -1, heads, C // heads).transpose(1, 2)
    vv = lin(ctx, k + ".to_v").reshape(Bn, -1, heads, C // heads).transpose(1, 2)
    s = (q @ kk.transpose(-1, -2)) / np.sqrt(np.float32(C // heads))
    o = (torch.softmax(s, -1) @ vv).transpose(1, 2).reshape(Bn, P, C)
    o = lin(o, k + ".to_out.0")
    return o.transpose(1, 2).reshape(Bn, C, H, Wd)

def transformer(x, k, heads, ctx):
    h = gn(x, k + ".norm1", 1e-6)
    x = x + attn(h, k + ".attn1", heads)
    h = gn(x, k + ".norm2", 1e-6)
    x = x + attn(h, k + ".attn2", heads, ctx)
    h = gn(x, k + ".norm3", 1e-6)
    Bn, C, H, Wd = h.shape
    h = h.reshape(Bn, C, H * Wd).transpose(1, 2)
    g = lin(h, k + ".ff.net.0.proj")
    a, b = g.chunk(2, -1)
    h = a * b * torch.sqrt(torch.tensor(0.5)) * (1 + torch.erf(b / torch.sqrt(torch.tensor(2.0))))
    h = lin(h, k + ".ff.net.2")
    return x + h.transpose(1, 2).reshape(Bn, C, H, Wd)

def resblock(x, k, temb):
    cin, cout = x.shape[1], W(k + ".conv2").shape[0]
    h = silu(gn(x, k + ".norm1", 1e-5))
    h = conv3(h, k + ".conv1")
    h = h + lin(temb, k + ".time_emb_proj").reshape(1, -1, 1, 1)
    h = silu(gn(h, k + ".norm2", 1e-5))
    h = conv3(h, k + ".conv2")
    if cin != cout:
        x = F.conv2d(x, W(k + ".skip_connection"), B(k + ".skip_connection"))
    return x + h

def silu(x):
    return x * torch.sigmoid(x)

lat = torch.from_numpy(np.fromfile(lat_path, dtype="<f4").copy().reshape(1, 4, 64, 64))
ctx = torch.from_numpy(np.fromfile(ctx_path, dtype="<f4").copy().reshape(1, 77, 2048))
pooled = torch.from_numpy(np.fromfile(pooled_path, dtype="<f4").copy())

half = 160
freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
arg = t * freqs
emb = torch.cat([torch.cos(arg), torch.sin(arg)]).reshape(1, -1)
temb = lin(silu(lin(emb, "time_embed.0")), "time_embed.2")
cat = torch.cat([pooled.reshape(1, -1), torch.zeros(1, 1536)], 1)
a = lin(silu(lin(cat, "label_emb.0.0")), "label_emb.0.2")
temb = temb + a

x = conv3(lat, "input_blocks.0.0")
hs = [(x, 320)]
def dm(k, attn_layers, heads):
    global x
    x = resblock(x, k + ".0", temb)
    if attn_layers:
        x = transformer(x, k + ".1", heads, ctx)
    hs.append((x, x.shape[1]))

dm("input_blocks.1", 0, 0)
dm("input_blocks.2", 0, 0)
x = conv3(x, "input_blocks.3.0.op", 2)
hs.append((x, x.shape[1]))
dm("input_blocks.4", 2, 5)
dm("input_blocks.5", 2, 5)
x = conv3(x, "input_blocks.6.0.op", 2)
hs.append((x, x.shape[1]))
dm("input_blocks.7", 10, 10)
dm("input_blocks.8", 10, 10)

x = resblock(x, "middle_block.0", temb)
x = transformer(x, "middle_block.1", 10, ctx)
x = resblock(x, "middle_block.2", temb)

for j in range(9):
    s, cs = hs.pop()
    x = torch.cat([x, s], 1)
    tl = 10 if j < 3 else (2 if j < 6 else 0)
    heads = 10 if j < 3 else 5
    k = f"output_blocks.{j}"
    x = resblock(x, k + ".0", temb)
    if tl:
        x = transformer(x, k + ".1", heads, ctx)
    upk = ".2.conv" if k + ".2.conv.weight" in T else (".1.conv" if k + ".1.conv.weight" in T else None)
    if upk:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = conv3(x, k + upk)

x = silu(gn(x, "out.0", 1e-5))
x = conv3(x, "out.2")
eps = x.reshape(-1).numpy()
print(f"ref eps rms {np.sqrt((eps.astype(np.float64)**2).mean()):.5f}")
eps.tofile("ref_eps.f32")
