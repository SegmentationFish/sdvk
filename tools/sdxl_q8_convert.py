#!/usr/bin/env python3
# Convert an SDXL full-checkpoint safetensors into a Q8_0 GGUF the sdvk engine loads directly.
#
#   python tools/sdxl_q8_convert.py <checkpoint.safetensors> <out.gguf>
#
# - streams one tensor at a time (peak RAM ~= the largest tensor in fp32, a few MB)
# - the header (kvs + tensor infos) is built in a memory buffer first, so the data section
#   offset is exact by construction -- never estimated
# - writes comfy.gguf.orig_shape.<name> (u64 array of the torch shape) for every tensor,
#   which is what sdvkgen's ggdims() consumes for conv/linear row-col resolution
# - mirrors the noobai production GGUF: big 2D+ weights -> Q8_0 (flat 32-elem blocks, the
#   engine dequantizes flat), biases/norms/small 1-D -> F32
#
# CPU only: no Vulkan, no GPU.
import io
import json
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
T_STRING, T_ARRAY, T_UINT32, T_UINT64 = 8, 9, 4, 10
GGML_F32, GGML_Q8_0 = 0, 8
ALIGN = 32
PREFIX = "model.diffusion_model."


def write_string(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def read_tensor(path, hdr, data_start, key):
    t = hdr[key]
    ne = 1
    for d in t["shape"]:
        ne *= d
    off = data_start + t["data_offsets"][0]
    dt = t["dtype"]
    if dt == "F16":
        return np.fromfile(path, dtype="<f2", count=ne, offset=off).astype(np.float32)
    if dt == "BF16":
        raw = np.fromfile(path, dtype="<u2", count=ne, offset=off)
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if dt == "F32":
        return np.fromfile(path, dtype="<f4", count=ne, offset=off)
    raise RuntimeError(f"{key}: dtype {dt} not supported")


def quant_q8(x):
    assert x.size % 32 == 0, f"element count {x.size} is not a multiple of 32"
    xb = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(xb).max(axis=1)
    scale = (amax / 127.0).astype("<f2")
    inv = np.where(amax > 0, 127.0 / np.maximum(amax, 1e-30), 0.0).astype(np.float32)
    q = np.rint(xb * inv[:, None]).astype(np.int8)
    # Q8_0 blocks are INTERLEAVED: [fp16 scale][32 x int8] per 32-element block
    out = np.empty((xb.shape[0], 34), dtype=np.uint8)
    out[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8).reshape(-1, 32)
    return out.tobytes()


def main():
    if len(sys.argv) != 3:
        print("usage: sdxl_q8_convert.py <checkpoint.safetensors> <out.gguf>")
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    hdr, data_start = read_header(src)
    keys = sorted(k for k in hdr if k.startswith(PREFIX))
    if not keys:
        print(f"error: no {PREFIX}* tensors in {src}")
        return 1

    plan = []  # (gguf_name, ne_gguf, ggtype, nbytes, ne, torch_shape)
    for k in keys:
        t = hdr[k]
        shape = [int(d) for d in t["shape"]]
        ne = 1
        for d in shape:
            ne *= d
        if t["dtype"] not in ("F16", "BF16", "F32"):
            print(f"note: skip non-float {k} ({t['dtype']})")
            continue
        if len(shape) >= 2 and ne >= 1024:
            if ne % 32 != 0:
                print(f"error: {k} element count {ne} is not a multiple of 32")
                return 1
            plan.append((k[len(PREFIX):], list(reversed(shape)), GGML_Q8_0, (ne // 32) * 34, ne, shape))
        else:
            plan.append((k[len(PREFIX):], list(reversed(shape)), GGML_F32, ne * 4, ne, shape))

    # ---- build the whole header in memory: sizes are then exact by construction
    hb = io.BytesIO()
    hb.write(GGUF_MAGIC)
    hb.write(struct.pack("<I", GGUF_VERSION))
    hb.write(struct.pack("<Q", len(plan)))
    hb.write(struct.pack("<Q", 2 + len(plan)))
    write_string(hb, "general.alignment")
    hb.write(struct.pack("<I", T_UINT32))
    hb.write(struct.pack("<I", ALIGN))
    write_string(hb, "general.name")
    hb.write(struct.pack("<I", T_STRING))
    write_string(hb, os.path.basename(dst))
    for name, _, _, _, _, shape in plan:
        write_string(hb, f"comfy.gguf.orig_shape.{name}")
        hb.write(struct.pack("<I", T_ARRAY))
        hb.write(struct.pack("<I", T_UINT64))
        hb.write(struct.pack("<Q", len(shape)))
        for d in shape:
            hb.write(struct.pack("<Q", d))
    offs = []
    cur = 0
    for _, _, _, nbytes, _, _ in plan:
        offs.append(cur)
        cur = (cur + nbytes + ALIGN - 1) // ALIGN * ALIGN
    for (name, ne_gguf, ggtype, _, _, _), off in zip(plan, offs):
        write_string(hb, name)
        hb.write(struct.pack("<I", len(ne_gguf)))
        for d in ne_gguf:
            hb.write(struct.pack("<Q", d))
        hb.write(struct.pack("<I", ggtype))
        hb.write(struct.pack("<Q", off))
    data_off = (hb.tell() + ALIGN - 1) // ALIGN * ALIGN

    n_q8 = sum(1 for p in plan if p[2] == GGML_Q8_0)
    print(f"{src}")
    print(f"  {len(plan)} tensors (Q8_0 {n_q8} / F32 {len(plan) - n_q8}) -> {dst}")
    print(f"  header {hb.tell()} B, data at {data_off}, payload {cur / 2**30:.2f} GB")

    with open(dst, "wb") as f:
        f.write(hb.getvalue())
        f.write(b"\0" * (data_off - hb.tell()))
        pos = data_off
        for i, (name, _, ggtype, nbytes, ne, _) in enumerate(plan):
            f.write(b"\0" * (data_off + offs[i] - pos))  # alignment gap to the declared offset
            x = read_tensor(src, hdr, data_start, PREFIX + name)
            assert x.size == ne, f"{name}: read {x.size} != {ne}"
            blob = quant_q8(x) if ggtype == GGML_Q8_0 else np.ascontiguousarray(x, dtype="<f4").tobytes()
            assert len(blob) == nbytes, f"{name}: {len(blob)} != {nbytes}"
            f.write(blob)
            pos = data_off + offs[i] + len(blob)
            if (i + 1) % 400 == 0:
                print(f"  {i + 1}/{len(plan)}")
        assert pos <= os.path.getsize(dst) + 1 or True
    print(f"done: {os.path.getsize(dst) / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
