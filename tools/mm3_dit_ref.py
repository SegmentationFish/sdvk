"""Official-reference golden dumper for the MiniMax Music 3 synthesis stack."""
import re
import sys

import numpy as np
import torch
import json
import struct

sys.path.insert(0, r"D:\diffusers\src")
from diffusers.models.condition_embedders.condition_embedder_minimax_music3 import (
    MiniMaxMusic3ConditionEncoder,
)
from diffusers.models.transformers.transformer_minimax_music3 import (
    MiniMaxMusic3Transformer1DModel,
)

CKPT, OUT = sys.argv[1], sys.argv[2]
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 16
LATENT = int(sys.argv[4]) if len(sys.argv) > 4 else 86
T = 0.6
NLAYERS = int(__import__("os").environ.get("MM3_REF_LAYERS", "2"))  # 36 does not fit in 16 GB RAM


def put(nm, t):
    a = np.ascontiguousarray(np.asarray(t.detach().float().cpu()).ravel())
    a.tofile(f"{OUT}/{nm}.f32")


class RawST:
    """Minimal reader: safetensors 0.8.0 crashed on this 4.6 GB fp16 file."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.base = 8 + n

    def keys(self):
        return [k for k in self.hdr if k != "__metadata__"]

    def shape(self, k):
        return self.hdr[k]["shape"]

    def tensor(self, k):
        m = self.hdr[k]
        o0, o1 = m["data_offsets"]
        self.f.seek(self.base + o0)
        buf = self.f.read(o1 - o0)
        dt = m["dtype"]
        if dt == "F16":
            a = np.frombuffer(buf, dtype="<f2").astype(np.float32)
        elif dt == "BF16":
            u = np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16
            a = u.view(np.float32)
        else:
            a = np.frombuffer(buf, dtype="<f4").astype(np.float32)
        return a.reshape(m["shape"])


fk = RawST(CKPT)
keys = fk.keys()
print("tensors", len(keys), flush=True)

COND = {"cond_layer_logits": "layer_weight_logits", "cond_layer_scale": "layer_scale"}
LAYER = [
    ("pre_norm.weight", "norm1.weight"), ("pre_norm.beta", "norm1.bias"),
    ("pre_norm.gamma", "norm1.weight"),
    ("ff_norm.weight", "norm2.weight"), ("ff_norm.beta", "norm2.bias"),
    ("ff_norm.gamma", "norm2.weight"),
    ("self_attn.to_out.weight", "attn.to_out.0.weight"),
    ("ff.ff.0.proj.weight", "ff_in.weight"), ("ff.ff.0.proj.bias", "ff_in.bias"),
    ("ff.ff.2.weight", "ff_out.weight"), ("ff.ff.2.bias", "ff_out.bias"),
]
TOP = {
    "diffusion_transformer.preprocess_conv.weight": "preprocess_conv.weight",
    "diffusion_transformer.postprocess_conv.weight": "postprocess_conv.weight",
    "diffusion_transformer.transformer.project_in.weight": "proj_in.weight",
    "diffusion_transformer.transformer.project_out.weight": "proj_out.weight",
    "diffusion_transformer.timestep_features.weight": "time_proj.weight",
    "diffusion_transformer.to_timestep_embed.0.weight": "time_embed.linear_1.weight",
    "diffusion_transformer.to_timestep_embed.0.bias": "time_embed.linear_1.bias",
    "diffusion_transformer.to_timestep_embed.2.weight": "time_embed.linear_2.weight",
    "diffusion_transformer.to_timestep_embed.2.bias": "time_embed.linear_2.bias",
}

torch.set_default_dtype(torch.float16)
cond_enc = MiniMaxMusic3ConditionEncoder().eval()
dit = MiniMaxMusic3Transformer1DModel(num_layers=NLAYERS).eval()
torch.set_default_dtype(torch.float32)
DPAR = dict(dit.named_parameters())
CPAR = dict(cond_enc.named_parameters())


def copy_in(params, dst_name, src_name, split=None):
    if dst_name not in params:
        raise KeyError(dst_name + " not in reference model")
    with torch.no_grad():
        src = torch.from_numpy(fk.tensor(src_name))
        if split is not None:
            src = src.chunk(3, dim=0)[split]
        dst = params[dst_name]
        if list(src.shape) != list(dst.shape):
            raise ValueError("shape mismatch %s %s -> %s %s" % (src_name, tuple(src.shape), dst_name, tuple(dst.shape)))
        dst.copy_(src)
    del src


n_loaded = 0
for k in keys:
    if k in TOP:
        copy_in(DPAR, TOP[k], k)
        n_loaded += 1
    elif k in COND:
        copy_in(CPAR, COND[k], k)
        n_loaded += 1
    elif k.startswith("latent_conditioners.0."):
        copy_in(CPAR, "proj." + k.split(".")[-1], k)
        n_loaded += 1
    else:
        m = re.match(r"diffusion_transformer\.transformer\.layers\.(\d+)\.(.+)", k)
        if not m:
            continue
        idx, rest = int(m.group(1)), m.group(2)
        if idx >= NLAYERS:
            continue
        base = "transformer_blocks.%d." % idx
        if rest == "self_attn.to_qkv.weight":
            for i, nm in enumerate(("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight")):
                copy_in(DPAR, base + nm, k, split=i)
            n_loaded += 1
            continue
        for suffix, dst in LAYER:
            if rest == suffix:
                copy_in(DPAR, base + dst, k)
                n_loaded += 1
                break
print("mapped tensors", n_loaded, flush=True)

g = torch.Generator().manual_seed(7)
cond_in = torch.randn(1, FRAMES, 8 * 4096, generator=g).half()
with torch.no_grad():
    cond_out = cond_enc(cond_in)
print("cond_out", tuple(cond_out.shape), flush=True)
put("cond_in", cond_in)
put("cond_out", cond_out)

x = torch.randn(1, 128, LATENT, generator=g).half()
t = torch.tensor([T], dtype=torch.float16)
blocks = {}
hooks = [b.register_forward_hook(lambda m, i, o, i2=i: blocks.__setitem__(i2, o))
         for i, b in enumerate(dit.transformer_blocks)]
probes = {}
hooks.append(dit.transformer_blocks[0].register_forward_pre_hook(
    lambda m, i: probes.__setitem__("pre", i[0])))
hooks.append(dit.transformer_blocks[0].attn.register_forward_hook(
    lambda m, i, o: probes.__setitem__("attn", o)))
hooks.append(dit.transformer_blocks[0].ff_out.register_forward_hook(
    lambda m, i, o: probes.__setitem__("ff", o)))
hooks.append(dit.transformer_blocks[0].norm1.register_forward_hook(
    lambda m, i, o: probes.__setitem__("norm1", o)))
hooks.append(dit.transformer_blocks[0].attn.to_out[0].register_forward_pre_hook(
    lambda m, i: probes.__setitem__("pv", i[0])))
for nm in ("to_q", "to_k", "to_v"):
    hooks.append(getattr(dit.transformer_blocks[0].attn, nm).register_forward_hook(
        lambda m, i, o, nm=nm: probes.__setitem__(nm, o)))
with torch.no_grad():
    v = dit(x, t, cond_out).sample
for h in hooks:
    h.remove()
print("dit out", tuple(v.shape), "rms %.4f" % float(v.float().pow(2).mean().sqrt()), flush=True)
put("dit_x", x)
put("dit_t", t)
put("dit_out", v)
for name, t in probes.items():
    put("ref_" + name, t)
for i in range(NLAYERS):
    if i in blocks:
        put("blk%d" % i, blocks[i])
print("ok")






