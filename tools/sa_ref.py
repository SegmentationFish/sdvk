"""Offline torch reference for Stable Audio Open Small (not part of the sdvk runtime).

Produces golden tensors for the native C++/Vulkan port, one subsystem at a time:

  python tools/sa_ref.py t5    --out DIR            # tokenizer ids + T5 last_hidden_state
  python tools/sa_ref.py dit   --out DIR            # one DiT forward (fixed x, t) + per-layer dumps
  python tools/sa_ref.py vae   --out DIR            # Oobleck decode of a fixed latent + per-layer dumps
  python tools/sa_ref.py gen   --out DIR            # 8-step pingpong sampling + audio

Every dump is raw little-endian f32 named <name>_<d0>x<d1>.f32 (1-D files omit the shape),
so the C++ side can load them with a tiny reader. `manifest.txt` lists name/shape/path.

The audio path uses stable-audio-tools 0.0.20 for the DiT/conditioning and diffusers'
AutoencoderOobleck for the decoder, i.e. the exact code the checkpoint was trained with.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = "E:/sdvkmodel/audio/stable-audio-small"
SR = 44100
DS = 2048  # downsampling ratio


# ------------------------------------------------------------------ dump helpers
class Dumper:
    def __init__(self, out):
        self.out = out
        self.lines = []
        os.makedirs(out, exist_ok=True)

    def f32(self, name, t):
        t = np.asarray(t.detach().cpu() if hasattr(t, "detach") else t)
        a = np.ascontiguousarray(np.asarray(t, dtype=np.float32).reshape(
            t.shape if t.ndim > 0 else (1,)))
        if a.ndim == 0:
            a = a.reshape(1)
        path = f"{name}_{'x'.join(str(d) for d in a.shape)}.f32" if a.ndim > 1 else f"{name}.f32"
        a.tofile(os.path.join(self.out, path))
        self.lines.append(f"{name} {' '.join(str(d) for d in a.shape)} {path}")
        return path

    def i32(self, name, t):
        a = np.ascontiguousarray(np.asarray(t, dtype=np.int32))
        path = f"{name}.i32"
        a.tofile(os.path.join(self.out, path))
        self.lines.append(f"{name} {' '.join(str(d) for d in a.shape)} {path}")
        return path

    def close(self):
        with open(os.path.join(self.out, "manifest.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def write_wav(path, audio, sr=SR):
    """audio: (channels, samples) float32 in [-1, 1] -> 16-bit PCM wav."""
    a = np.clip(np.asarray(audio, dtype=np.float32).T, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    n, ch = pcm.shape
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(int(36 + n * ch * 2).to_bytes(4, "little"))
        f.write(b"WAVEfmt ")
        f.write((16).to_bytes(4, "little"))
        f.write((1).to_bytes(2, "little"))  # PCM
        f.write(ch.to_bytes(2, "little"))
        f.write(sr.to_bytes(4, "little"))
        f.write((sr * ch * 2).to_bytes(4, "little"))
        f.write((ch * 2).to_bytes(2, "little"))
        f.write((16).to_bytes(2, "little"))
        f.write(b"data")
        f.write((n * ch * 2).to_bytes(4, "little"))
        f.write(pcm.tobytes())


# ------------------------------------------------------------------ T5
def t5_dump(args, d):
    from transformers import AutoTokenizer, T5EncoderModel

    path = f"{ROOT}/t5-base"
    tok = AutoTokenizer.from_pretrained(path)
    enc = tok(
        [args.prompt],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    model = T5EncoderModel.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask).last_hidden_state
    print("ids", ids.shape, ids[0, :12].tolist())
    print("last_hidden", tuple(out.shape), "rms %.4f" % float(out.pow(2).mean().sqrt()))
    d.i32("t5_ids", ids[0].numpy())
    d.i32("t5_mask", mask[0].numpy())
    d.f32("t5_last", out[0].numpy())
    d.close()


# ------------------------------------------------------------------ model loader
def load_dit():
    """Build the DiffusionTransformer alone and load model.model.* from the checkpoint.

    Avoids the T5 conditioner (~900 MB) that create_model_from_config would pull in, which
    matters on this 16 GB box: with the DiT and the Oobleck decoder resident there is no room
    for it, and torch died inside Linear.__init__ with an access violation.
    """
    from stable_audio_tools.models.dit import DiffusionTransformer

    with open(f"{ROOT}/model_config.json", encoding="utf-8") as f:
        cfg = json.load(f)["model"]["diffusion"]["config"]
    dit = DiffusionTransformer(
        io_channels=cfg["io_channels"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        cond_token_dim=cfg["cond_token_dim"],
        project_cond_tokens=True,
        global_cond_dim=cfg["global_cond_dim"],
        project_global_cond=True,
        transformer_type="continuous_transformer",
        diffusion_objective="rf_denoiser",
        attn_kwargs={"qk_norm": "ln"},
    )
    state = load_file(f"{ROOT}/model.safetensors")
    sub = {k[len("model.model."):]: v for k, v in state.items() if k.startswith("model.model.")}
    missing, unexpected = dit.load_state_dict(sub, strict=False)
    print(f"dit load: {len(sub)} tensors, missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:6])
    if unexpected:
        print("  unexpected:", unexpected[:6])
    return dit.eval().float()


def load_model():
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models import conditioners as sat_cond

    with open(f"{ROOT}/model_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    local_t5 = f"{ROOT}/t5-base"
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["id"] == "prompt":
            c["config"]["t5_model_name"] = local_t5
    if local_t5 not in sat_cond.T5Conditioner.T5_MODELS:
        sat_cond.T5Conditioner.T5_MODELS.append(local_t5)
        sat_cond.T5Conditioner.T5_MODEL_DIMS[local_t5] = 768

    model = create_model_from_config(cfg)
    state = load_file(f"{ROOT}/model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:6])
    model = model.eval().float()
    return model, cfg


def conditioning(model, prompt, seconds):
    """Returns (cross_attn_cond [1,65,768], global_cond [1,768]) through the real conditioner."""
    cond = model.conditioner([{"prompt": prompt, "seconds_total": seconds}], device="cpu")
    ids = model.cross_attn_cond_ids
    xattn = torch.cat([cond[k][0] for k in ids], dim=1) if len(ids) > 1 else cond[ids[0]][0]
    gids = model.global_cond_ids
    g = torch.cat([cond[k][0] for k in gids], dim=1) if len(gids) > 1 else cond[gids[0]][0]
    g = g.reshape(g.shape[0], -1)  # global conditioning is [B, dim]; the conditioner emits [B, 1, dim]
    return cond, xattn, g


# ------------------------------------------------------------------ DiT
def dit_dump(args, d):
    model, cfg = load_model()
    wrapper = model.model
    dit = wrapper.model
    cond, xattn, g = conditioning(model, args.prompt, args.seconds)
    print("cond tokens", tuple(xattn.shape), "global", tuple(g.shape))
    for k in cond:
        print("  ", k, tuple(cond[k][0].shape))

    T = args.frames
    gtorch = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, 64, T, generator=gtorch)
    t = torch.tensor([args.t])
    with torch.no_grad():
        y = wrapper(x, t, cross_attn_cond=xattn, global_cond=g)
    print("dit out", tuple(y.shape), "rms %.4f" % float(y.pow(2).mean().sqrt()))

    d.f32("dit_x", x[0])
    d.f32("dit_t", np.array([args.t], dtype=np.float32))
    d.f32("cond_tokens", xattn[0])
    d.f32("sec_embed", g[0])
    d.f32("dit_out", y[0])
    # embedding stages the port needs to reproduce
    with torch.no_grad():
        ce = dit.to_cond_embed(xattn)[0]
        ge = dit.to_global_embed(g)[0]
        te = dit.to_timestep_embed(dit.timestep_features(t[:, None].float()))[0]
        pre = dit.preprocess_conv(x)
        gsum = ge + te
    d.f32("cond_proj", ce)
    d.f32("global_proj", ge)
    d.f32("time_proj", te)
    d.f32("global_time_sum", gsum)
    d.f32("preproc_out", pre[0])
    d.close()


# ------------------------------------------------------------------ VAE
def vae_dump(args, d):
    from diffusers import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(f"{ROOT}/vae", torch_dtype=torch.float32).eval()
    L = args.frames
    g = torch.Generator().manual_seed(args.seed)
    z = torch.randn(1, 64, L, generator=g)
    with torch.no_grad():
        aud = vae.decode(z).sample
    print("vae out", tuple(aud.shape), "peak %.4f" % float(aud.abs().max()))
    d.f32("vae_z", z[0])
    d.f32("vae_audio", aud[0])
    d.close()
    write_wav(os.path.join(d.out, "vae.wav"), aud[0].numpy())


# ------------------------------------------------------------------ full generation
def gen_dump(args, d):
    dit = load_dit()
    cond_dir = args.cond_from or args.out
    if args.cond_from:
        # reuse the cond tensors from a previous `dit` dump: avoids loading T5 a second time
        xattn = np.fromfile(f"{cond_dir}/cond_tokens_65x768.f32", dtype="<f4").reshape(1, 65, 768)
        g = np.fromfile(f"{cond_dir}/sec_embed.f32", dtype="<f4").reshape(1, 768)
        xattn, g = torch.from_numpy(xattn.copy()), torch.from_numpy(g.copy())
        print("conditioning from", cond_dir)
    else:
        model, _ = load_model()
        cond, xattn, g = conditioning(model, args.prompt, args.seconds)
        del model
    sample_size = int(args.seconds * SR) // DS * DS
    T = sample_size // DS
    print(f"latent frames {T} ({sample_size} samples)")

    torch.manual_seed(args.seed)
    x = torch.randn(1, 64, T)
    d.f32("gen_noise", x[0])

    steps = args.steps
    ts = torch.linspace(1.0, 0.0, steps + 1)
    latents = []
    with torch.no_grad():
        for i in range(steps):
            tc, tn = ts[i], ts[i + 1]
            v = dit(x, tc.repeat(1), cross_attn_cond=xattn, global_cond=g)
            denoised = x - tc * v
            x = (1 - tn) * denoised + tn * torch.randn_like(x)
            latents.append(x[0].clone())
            print(f"step {i + 1}/{steps} t {float(tc):.4f} -> {float(tn):.4f}  "
                  f"|x| {float(x.norm()):.3f}")
    d.f32("gen_latent", x[0])
    for i, l in enumerate(latents):
        d.f32(f"gen_step{i + 1}", l)
    d.f32("gen_ts", ts)

    from diffusers import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(f"{ROOT}/vae", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        aud = vae.decode(x.to(torch.float32)).sample[0].numpy()
    print("audio", aud.shape, "peak %.4f rms %.4f" % (np.abs(aud).max(), np.sqrt((aud ** 2).mean())))
    write_wav(os.path.join(d.out, "gen.wav"), aud)
    d.f32("gen_audio", aud)
    d.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["t5", "dit", "vae", "gen"])
    ap.add_argument("--out", default="D:/tmp/saref")
    ap.add_argument("--prompt", default="Lo-fi hip hop instrumental with vinyl crackle and gentle piano.")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=43, help="latent frames for dit/vae dumps")
    ap.add_argument("--t", type=float, default=0.7, help="timestep for the dit dump")
    ap.add_argument("--cond-from", default="", help="reuse cond_tokens/sec_embed from this dir")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SA_THREADS", os.cpu_count() or 4)))
    d = Dumper(args.out)
    {"t5": t5_dump, "dit": dit_dump, "vae": vae_dump, "gen": gen_dump}[args.mode](args, d)


if __name__ == "__main__":
    main()
