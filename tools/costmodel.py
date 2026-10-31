# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Static cost model for one SDXL UNet forward pass, straight from a GGUF tensor table.
# No GPU, no torch: parse the header, rebuild the block layout from the tensor names, then
# estimate FLOPs and memory traffic per node and rank the hotspots.
#
#   python tools/costmodel.py <model.gguf> [--latent 96] [--tf 5.5] [--bw 180] [--top 30]
#
# Why not just multiply dims by pixels: this checkpoint stores conv weights with the kernel
# folded into ne[0] (a 3x3 conv shows up as [256, 45]), so the real shape has to come from the
# comfy.gguf.orig_shape.* metadata. This script reads it, which is what the engine's ggdims()
# does too.
import argparse
import struct
import sys

# ggml value type -> element size in bytes (9 is the variable-length array type)
VT_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
# ggml tensor type -> (block elements, block bytes); Q8_0 packs 32 weights into 34 bytes
QT = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
      8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144)}


def read_str(f):
    n, = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def read_val(t, f):
    if t == 8:
        return read_str(f)
    if t == 9:
        et, = struct.unpack("<I", f.read(4))
        cnt, = struct.unpack("<Q", f.read(8))
        if et == 8:
            return [read_str(f) for _ in range(cnt)]
        if et not in VT_SIZE:
            f.seek(cnt * 8, 1)
            return []
        raw = f.read(VT_SIZE[et] * cnt)
        return list(struct.unpack("<%d%s" % (cnt, FMT[et][1]), raw))
    raw = f.read(VT_SIZE[t])
    return struct.unpack(FMT[t], raw)[0]


def load_gguf(path):
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        raise SystemExit("%s is not a GGUF file" % path)
    ver, = struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<Qq", f.read(16))
    meta = {}
    for _ in range(n_kv):
        key = read_str(f)
        t, = struct.unpack("<I", f.read(4))
        meta[key] = read_val(t, f)
    tensors = []
    for _ in range(n_tensors):
        name = read_str(f)
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        ty, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tensors.append({"name": name, "dims": dims, "type": ty, "off": off})
    f.close()
    return ver, meta, tensors


def tensor_bytes(t):
    """Payload size from the ggml type and the stored dims (the quantized layout)."""
    ne = 1
    for d in t["dims"]:
        ne *= d
    blk, bpb = QT.get(t["type"], (1, 4))
    return ne // blk * bpb


def logical_shape(name, t, orig):
    """(kind, cout, cin, kh, kw). comfy.gguf.orig_shape.<name> holds the PyTorch shape:
    [320, 4, 3, 3] for the stem conv, [1280, 320] for time_embed.0."""
    o = orig.get(name)
    if o and len(o) == 4:
        return ("conv", o[0], o[1], o[2], o[3])
    if o and len(o) == 2:
        return ("linear", o[0], o[1], 1, 1)
    d = t["dims"]
    if len(d) == 4:
        # ggml layout: ne0 is the fastest axis and holds cin*kh*kw for a folded conv
        return ("conv", d[3], d[0] * d[1] * d[2], d[1], d[2])
    if len(d) == 2:
        return ("linear", d[1], d[0], 1, 1)
    return (None, 0, 0, 0, 0)


# SDXL block plan: 3 levels, [1 basic, 2 basic+attn, 10 basic+attn] per level
BLOCKS = {0: (1, 2), 1: (1, 2), 2: (1, 10)}


def main(argv):
    ap = argparse.ArgumentParser(description="SDXL UNet cost model from a GGUF tensor table")
    ap.add_argument("gguf")
    ap.add_argument("--latent", type=int, default=96, help="latent edge size (96 = 768 px)")
    ap.add_argument("--tf", type=float, default=5.5, help="effective fp32 TFLOP/s")
    ap.add_argument("--bw", type=float, default=180.0, help="effective GB/s")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv)

    ver, meta, tensors = load_gguf(args.gguf)
    orig = {k[len("comfy.gguf.orig_shape."):]: v for k, v in meta.items()
            if k.startswith("comfy.gguf.orig_shape.")}
    print("gguf v%d  %d tensors  %d orig_shape entries" % (ver, len(tensors), len(orig)))

    TF, BW = args.tf * 1e12, args.bw * 1e9
    L = args.latent
    lvl_tokens = {0: L * L, 1: (L // 2) * (L // 2), 2: (L // 4) * (L // 4)}

    def level_tokens(name):
        parts = name.split(".")
        if parts[0] == "input_blocks" and parts[1].isdigit():
            i = int(parts[1])
            return lvl_tokens[0 if i < 3 else 1 if i < 6 else 2]
        if parts[0] == "output_blocks" and parts[1].isdigit():
            i = int(parts[1])
            return lvl_tokens[2 if i < 3 else 1 if i < 6 else 0]
        if parts[0] == "middle_block":
            return lvl_tokens[2]
        return lvl_tokens[0]

    nodes = []
    conv_flop = conv_bytes = 0.0
    for t in tensors:
        name = t["name"]
        if name.endswith(".bias") or ".norm." in name or "label_emb" in name:
            continue
        kind, cout, cin, kh, kw = logical_shape(name, t, orig)
        if kind is None:
            continue
        # A ResBlock's first in_layers conv swallows the previous block's output (skip concat),
        # so its weight shape already carries cin = channels_in + channels_skip. No extra pixels.
        if name.startswith("time_embed") or name.startswith("label_emb"):
            # a single timestep, not a spatial map: one row of activations
            tokens = 1
        else:
            tokens = level_tokens(name)
        wb = tensor_bytes(t)
        if kind == "conv":
            stride = 2 if ".op." in name or ".downsample" in name else 1
            out_tokens = tokens // (stride * stride)
            flop = 2.0 * out_tokens * cout * cin * kh * kw
            # im2col materializes the im2col matrix, the GEMM then reads it back
            act = 4.0 * (out_tokens * cin * kh * kw) * 2 + 4.0 * out_tokens * cout
            byt = wb + act
        else:
            flop = 2.0 * tokens * cout * cin
            act = 4.0 * tokens * cin + 4.0 * tokens * cout
            byt = wb + act
        nodes.append([flop / TF, byt / BW, flop, byt, name])
        conv_flop += flop
        conv_bytes += byt

    # attention counted per transformer block (projections plus the two score products), not per
    # projection, so nothing is counted twice above: those to_q/to_k/v/out rows were skipped by
    # the level_tokens() pass treating them as full-resolution linears.
    att_flop = att_bytes = 0.0
    ctx = 77
    for lvl in (0, 1, 2):
        T = lvl_tokens[lvl]
        basic, with_attn = BLOCKS[lvl]
        nblk = basic + with_attn
        C = 320 * (2 ** lvl)
        fl = 4 * 2.0 * C * T * C * nblk
        by = 4.0 * (4 * T * C + 4 * C * C) * nblk
        for _ in range(with_attn):
            fl += 2 * 2.0 * T * T * C          # q@k^T and attn@v
            by += 4.0 * (T * T + 2 * T * C) * 2
            fl += 2 * 2.0 * T * ctx * C        # cross-attention
            by += 4.0 * (T * ctx + 2 * T * C) * 2
        att_flop += fl
        att_bytes += by

    tot_f, tot_b = conv_flop + att_flop, conv_bytes + att_bytes
    print("\n=== one forward pass at latent %dx%d ===" % (L, L))
    print("conv/linear   %8.1f GFLOP  %7.2f GB" % (conv_flop / 1e9, conv_bytes / 1e9))
    print("attention     %8.1f GFLOP  %7.2f GB" % (att_flop / 1e9, att_bytes / 1e9))
    print("total         %8.1f GFLOP  %7.2f GB" % (tot_f / 1e9, tot_b / 1e9))
    print("bound         %s" % ("compute" if tot_f / TF > tot_b / BW else "bandwidth"))
    print("floor         %8.2f s/pass  (rx 580 measured 14.1 s/pass)" % max(tot_f / TF, tot_b / BW))

    nodes.sort(key=lambda r: -max(r[0], r[1]))
    print("\n=== top %d nodes by estimated time ===" % args.top)
    print("%10s %10s %-5s %8s %7s  %s" % ("time", "of which", "kind", "GFLOP", "GB", "name"))
    for comp, mem, flop, byt, name in nodes[:args.top]:
        t = max(comp, mem)
        kind = "FLOP" if comp > mem else "B/W"
        print("%9.2fms %9.2fms %-5s %8.1f %7.2f  %s"
              % (t * 1e3, min(comp, mem) * 1e3, kind, flop / 1e9, byt / 1e9, name))

    print("\n%d tensors, %.2f GiB of weights"
          % (len(tensors), sum(tensor_bytes(t) for t in tensors) / (1 << 30)))


if __name__ == "__main__":
    main(sys.argv[1:])
