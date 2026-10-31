import json
import struct
import sys


def prefixes(path, n=8):
    with open(path, "rb") as f:
        hdr = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hdr))
    meta.pop("__metadata__", None)
    names = list(meta.keys())
    print("%s: %d tensors" % (path, len(names)))
    seen = []
    for name in names:
        p = name.split(".")[0]
        if p not in seen:
            seen.append(p)
    print("  top-level: %s" % seen[:n])
    for name in names[:4]:
        print("  e.g. %s %s" % (name, meta[name]["shape"]))


for p in sys.argv[1:]:
    prefixes(p)
