# SPDX-License-Identifier: GPL-3.0-only
# Compare sdvk's encoder stage dumps (.f32, from OFC_VAE_ENC_DUMP=1) with the torch reference
# (tools/refencode.py, REF_DUMP_DIR=<dir> writes .npy). Reports relative error per stage so the
# first stage that diverges is obvious instead of "the latent is off by something".
# usage: python tools/enc_dump_cmp.py <engine_dir> <ref_dir>
import sys, os, glob
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

eng, ref = sys.argv[1], sys.argv[2]
pairs = []
for path in sorted(glob.glob(os.path.join(eng, "dbg_enc_*.f32"))):
    tag = os.path.basename(path)[len("dbg_enc_"):-4]
    tag = tag.split("_", 1)[1] if "_" in tag else tag
    rp = os.path.join(ref, "ref_enc_" + tag + ".npy")
    if os.path.exists(rp):
        pairs.append((tag, path, rp))
    else:
        print("skip %-14s no reference" % tag)

print("%-14s %8s %10s %10s %10s" % ("stage", "n", "rel_l2", "max_abs", "ref_rms"))
worst = None
for tag, e, r in pairs:
    a = np.fromfile(e, dtype="<f4").astype(np.float64)
    b = np.load(r).astype(np.float64).ravel()
    if a.size != b.size:
        print("%-14s %8d %s" % (tag, a.size, "SIZE MISMATCH ref=%d" % b.size))
        continue
    d = a - b
    nb = np.linalg.norm(b)
    rel = np.linalg.norm(d) / (nb if nb > 1e-12 else 1.0)
    mx = np.abs(d).max()
    print("%-14s %8d %10.3e %10.3e %10.3e" % (tag, a.size, rel, mx, nb / np.sqrt(b.size)))
    if worst is None or rel > worst[1]:
        worst = (tag, rel)
if worst:
    print("\nfirst diverging judgement: worst stage %s rel_l2 %.3e" % worst)
