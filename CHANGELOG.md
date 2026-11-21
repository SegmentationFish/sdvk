# Changelog

All notable changes to sdvk. The project is pre-1.0; the format follows
[Keep a Changelog](https://keepachangelog.com/) and versions are approximate.

## 2026-11-21

### Fixed

- **VAE encode wrote past the 48 MB im2col buffer**: `im2col_s2` generated the whole stride-2 column
  block in one dispatch, so a 128-channel downsample at 640x512 needed 360 MB and the overflow landed in
  the scratch arena (visible as the arena growing 220 -> 700 MB). Everything computed afterwards read
  corrupted memory, which is why the encoded latent was nowhere near the reference. The shader and
  `conv3s2` now walk the output in row bands whose col block always fits, exactly like `conv3Raw`.
- **Encoder parked the input without copying it**: after `conv_in` the full-resolution tensor was
  "moved" to the top of the step arena by changing `BufRef.offset` only, so the first resnet read
  uninitialised arena memory. It is now a real `copyInto` + barrier.
- **img2img strength started at a far too noisy sigma**: `buildPartial` scaled `sigma_max` by the
  denoise strength (`sigma_max * 0.6` ~ 8.8 while the seed latent has rms ~1.1), burying the source
  image under noise; only the coarsest layout survived, no matter how many steps ran. It now follows
  diffusers and keeps the *tail* of the full schedule (`start = steps - round(steps * strength)`).
  Same seed/prompt/init at 20 steps went from "a red disc on black" to the original composition with
  the subject intact.
- `sampler euler:` now reports the actual first sigma (`sigma0`) instead of the schedule's global
  `sigma_max`, which was misleading for partial runs.

### Changed

- Removed the temporary encoder bisection probes (`OFC_ENC_PROBE`, `OFC_ENC_COLDUMP`, `OFC_ENC_WDUMP`,
  `OFC_ENC_RAWCOPY`, `OFC_ENC_IMPULSE`, `OFC_ENC_SCRATCH_FILL`). The diagnostics that stayed useful are
  kept: `OFC_LATENT_DUMP`, `OFC_VAE_ENC_DUMP` / `OFC_ENC_DUMP_TAG` / `OFC_ENC_DUMP_SMALL`,
  `OFC_ENC_SCRATCH_MB`, `OFC_ENC_NOREUSE`.
- `tools/stress.sh` roundtrip threshold 30 -> 28 dB. Encode->decode is lossy by itself (29-33 dB on
  natural artwork, 36 dB on a second-generation image), while a broken encoder reads 12-18 dB, so the
  threshold only has to separate those regimes.

### Verified

- `--roundtrip` 512x640 = 29.90 / 29.96 dB on two different sources, 768x1024 = 36.74 dB.
- Latent vs `tools/refencode.py` (torch, CPU) at 512x640: rel L2 0.0075, corr 0.99998.
- img2img 512x640, denoise 0.6, 20 steps: composition and subject preserved.
- inpaint 512x640, mask 4.1% repainted, feather 8: mean |diff| 15.0 inside the mask vs 3.2 outside
  (outside equals the VAE roundtrip loss, i.e. 30.5 dB against the resampled source).
- text2img 30 steps at 512x640 (29.9 -> clean) and 768x1024 (clean), so the earlier "mushy 30-step
  render" no longer reproduces.

## 2026-11-20

### Added

- **`bin/` launchers**: `sdvk`, `sdvkgen` and `sdvk-server` wrappers pin the absolute shader directory (the
  server also pins `webui/` and `output/`), so a PATH entry to `bin/` works from any cwd. The local user PATH
  now contains it.
- **`--meta <image.png>`**: prints the text metadata embedded in a generated PNG. Reads standard `tEXt`
  (keyword `sdvk`), `zTXt`, uncompressed/compressed `iTXt`, and the older keyword-less chunks written by
  earlier sdvk builds. No GPU or model is touched, so it also works while another job holds the card.
- **VAE-only `--decode`**: `--decode <latent.ofcl> --vae <file>` no longer requires a full model directory.
  `-m <dir>` auto-selects `vae/model.safetensors`, `vae/diffusion_pytorch_model.safetensors`,
  `vae.safetensors`, or the first `.safetensors` under `<dir>/vae/`. Decoded PNGs embed source, device,
  size and VAE name. Verified on the 832x1216 latent: explicit and auto-discovered VAE produced identical
  IDAT bytes (13.3 s tiled decode).
- New PNGs write metadata as a proper `tEXt` chunk (`sdvk\0` + block); old images without the keyword
  remain readable.
- **`--line <image.png> [-m <dir>]`** (alias `--cmd`): reconstructs a runnable `sdvk ...` command from the
  PNG metadata (model, prompt, negative, size, steps, cfg, seed, sampler, spacing, vpred/zsnr). The recorded
  model name is resolved recursively under `-m`, preferring `.gguf`; `--unet` overrides the lookup. The
  command goes to stdout, so it can be redirected or piped. LoRA, `--weights-host` and hires flags are not
  recorded. When vpred is present on a model without a vpred marker, stderr prints a note. Verified by
  executing the reconstructed 832x1216 command with small size/step overrides (exit 0).

### Fixed

- **v-prediction left on for epsilon models (ghost images)**: the web UI defaulted the v-pred
  checkbox to on. WAI-Illustrious and SDXL base are epsilon models; sampling them with v-pred
  collapses the latent to a flat wash (same-seed 8-step A/B: on = blank lavender, off = coherent
  portrait). The UI now defaults to eps and follows the selected UNet filename (`vpred`/`v-pred`
  turns it on automatically), so NoobAI vPred still checks itself. The corrected WAI
  832x1216/28-step run (`D:/tmp/wai_fixed_832x1216.png`, same seed/prompt/sampler, vpred off)
  renders normally.
- **Web UI scan cap**: the 4000-file guard counted every file, so a 15k-file library
  (`E:/sdvkmodel` with audio models) stopped before `models/` and hid the image UNets and text
  encoders. The cap now counts only `.gguf`/`.safetensors` candidates.

### Notes

- 832x1216 with resident Q8 weights needs a ~900 MB sampler arena that can fail on this 8 GB card
  when the desktop/browser already holds ~1 GB of VRAM. `--weights-host` streams the UNet from
  system RAM and completed the same seed (13.77 s/step, 424 s for 28 steps, tiled VAE 12.7 s);
  it needs several GB of free system RAM.

## 2026-11-14

### Added

- **K-quant GGUF support** (`Gguf.cpp`): Q2_K/Q3_K/Q4_K/Q5_K/Q6_K are dequantized with
  llama.cpp's 256-value super-block layouts and re-encoded to Q8_0 while the UNet loads, so
  the existing Q8 GEMM path runs them on an 8 GB card. Dequantization matches `gguf-py`
  bit-for-bit for all five types. `ggufinfo` and the arena planner size K-quant tensors as
  Q8 payload + scales instead of fp32.
- **`--unet-q8`**: UNet weights are encoded to Q8_0 at load (2D/4D tensors, 32-aligned),
  for safetensors checkpoints and for GGUF F16/BF16/F32 tensors. An fp32 SD1.5 UNet drops
  from 3.28 GB to 923 MB; the Hassaku Q8 GGUF drops from 3738 MB to 2759 MB (F16 tensors
  converted), which makes its 1024² plan fit the device (host RAM permitting).
- **Device queries / portable fast paths**: the engine now reads subgroup size and compute
  limits; fused attention falls back when subgroup <32, shared memory <16 KB or workgroups
  <256. `fused_attn_tiled` loads opportunistically and is skipped if pipeline creation fails
  (some drivers report a conservative 32 KB shared-memory limit but accept the 49,920-byte
  kernel). `SDVK_ATTN_TILED=0/1` overrides the choice.
- **Debug env hooks**: `OFC_PROBE_DUMP`, `OFC_PROBE_HASH_INPUTS`, `OFC_WEIGHT_HASH`,
  `OFC_DEV_WEIGHT_HASH`, `OFC_DEV_WEIGHT_DUMP`, `OFC_LORA_DUMP`, `OFC_CLEAR_STEP`.
- **VAE auto decode**: full-image decode for canvases up to 896² (768² VAE: **3.9 s vs 9.3 s**
  tiled), tiled above; `--vae-tile N` still overrides, `0` keeps the automatic choice.
- **Cross-attention K/V cache**: the 77-token context projections are computed once per run and
  reused for every sampler step and both CFG branches. Latent output is bit-identical; 768²
  cfg5 drops from 3.89 to 3.78 s/step. `OFC_NO_XKV=1` disables it for A/B tests.

### Fixed

- **SDXL real-text conditioning regression (critical).** An uncommitted text-encoder-GGUF
  change set CLIP-G's activation mode to 3, the CLIP-L GELU variant; CLIP-G needs mode 4.
  Every SDXL real-prompt run collapsed to texture while precomputed embeddings still worked.
  The tell was `ctx cond rms 0.958` instead of the validated `0.696`. Restored
  `actMode = wide ? 4 : 3`; WAI v170 768²/20-step/cfg5 now renders a normal illustration again,
  and the Q4_K_M build produces the same composition. The concurrent change that replaced the
  context with the final-layer-norm tensor was also reverted to the validated embOut.
- **Full-checkpoint text encoders and VAE.** Three separate bugs made a single-file SDXL
  checkpoint unusable even though the UNet loaded: CLIP-L was filtered by `Part::ClipL` on
  the original `text_encoder.` name (0 tensors uploaded), phase 1 re-loaded CLIP-L into the
  CLIP-G arena (OOM), and safetensors VAE keys kept the LDM `mid.block_1`/`decoder.up.N`
  layout (only the GGUF path mapped them). Fixed; original OpenCLIP bigG towers
  (`transformer.resblocks.*` with fused `in_proj`) are now converted to the diffusers
  `encoder.layers.*` layout on load, splitting q/k/v.
- **Weight upload corruption on the non-LoRA Q8 path.** `beginUpload` was called twice per
  tensor and the second reservation could flush the first, still-unwritten staging slot when
  the ring was near capacity. A boundary tensor
  (`input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight`) was uploaded from stale
  bytes; LoRA loads were accidentally correct because they used `stageUpload`. The loader
  now always reads Q8 blocks into host memory and uses `stageUpload`. LoRA scale 0 is now
  bit-identical to a no-LoRA probe, and device-side weight hashes match for all 1680 tensors.
- **Arena sizing for single-file checkpoints**: UNet and text-encoder arenas sum the tensors
  actually loaded instead of doubling the whole 4.26 GB file. SD1.5 CLIP-L arena: 2048 → 725 MB.
- **K-quant arena sizing**: a Q4_K_M file no longer under-allocates by ~750 MB.

### Verified

- WAI v170 Q4_K_M (578 Q4_K + 210 Q6_K + 892 F16) vs Q8_0: probe eps cos **0.99888**,
  rel 4.8%; 2-step 512² image mean |diff| **0.76/255**, corr **0.9984**.
- **JANKU v7.777** (6.94 GB SDXL full checkpoint, F16/BF16, 2515 tensors) loads directly with
  `--unet-q8`, using its embedded CLIP-L, converted bigG CLIP-G and embedded LDM VAE. 8-step
  512² eps/cfg5 output is a coherent lighthouse scene; `ctx cond rms 0.69523` differs from the
  shared CLIP run (0.68522), confirming the checkpoint's own text encoders are in use.
- **WAI v170 real-text production run**: 768², 20 steps, cfg 5, eps, euler/leading, Q8_0 +
  shared CLIP-L/G: **90.6 s total, 3.82 s/step, VAE 3.9 s**, normal purple-haired street
  illustration. The Q4_K_M build produces the same scene with a different pose/clothing.
- **Hassaku XL Lightning v0.6 Q8 GGUF** is usable after the upload fix: 512² is out of its
  1024² training distribution and still looks like kaleidoscope noise, but 768² with its
  recommended 4 steps / cfg 1.5 / eps produces a coherent image. Its 1024² run now allocates
  the 2950 MB Q8 arena and 1463 MB step arena but fails at the final 128 MB scratch
  allocation on this 16 GB workstation (only ~1.5 GB free system RAM, WDDM mirrors device
  memory); it should run once background RAM is freed. The earlier "UNet weights corrupted"
  diagnosis is obsolete.
- LoRA merge math (linear and conv/LoCon) matches a Python reference within **1.5e-8**.
  Probe eps at scale 0 is bit-identical to the base model after the upload fix. RICH_BG at
  0.3/0.5 is visually clean; full strength 1.0 at cfg 5 still produces texture, which tracks
  the base model (x3d works on NoobAI, not on WAI) rather than the merge.

### Known issues

- **SD1.5 sampling crashes inside `amdvlk64.dll`** after model load and sampler setup (first
  UNet step, hd=40/80 generic attention path). The SD1.5 adapter remains unverified.
- LoRA files still have no base-model metadata warning; DoRA is treated as plain LoRA.

## 2026-11-07

### Added

- **Full-checkpoint converter** (`tools/sdxl_q8_convert.py`): turns an SDXL full-checkpoint
  safetensors into the Q8_0 GGUF the engine loads (CPU-only, streams one tensor at a time,
  ~2.6 GB out). Mirrors the ComfyUI-GGUF conventions the loader expects: interleaved
  `[fp16 scale][32 x int8]` blocks over the flat tensor, `comfy.gguf.orig_shape.*` u64 arrays
  for every tensor, small biases/norms kept in F32. Round-trip error ~0.4%.
- **Multi-LoRA**: `--lora` is now repeatable and each spec accepts an inline strength
  (`--lora a.safetensors:0.8 --lora b.safetensors:0.5`). Deltas from all files merge
  cumulatively (`W += sum_i scale_i * (alpha_i/rank_i) * B_i @ A_i`); Q8 tensors are
  dequantized once, all deltas applied, then requantized once per row. The server API
  takes the same syntax as a comma-separated `"lora"` string, and `"loraScale"` still
  scales a single-path request. Fixed the resident-server behavior where a LoRA change
  between jobs was silently ignored: the LoRA set now participates in the reload decision
  alongside `modelDir`/`device`. `--lora-scale` keeps its old meaning (strength of the
  most recent `--lora`).

## 2026-11-06

### Added

- **Direct-prompt web studio** (`webui/`): model-directory scan, UNet/CLIP-L/CLIP-G/VAE/LoRA
  pickers, LoRA strength, per-step progress, cancel, explicit VRAM cleanup, and a
  latent → PNG card that runs the VAE alone. Settings persist in `localStorage`.
- **Server endpoints**: `GET /api/scan?dir=` returns per-component file lists and resolved
  defaults; `POST /api/decode` decodes a latent with the VAE only (unloading the UNet first);
  async job responses now include `"ok": true`.
- **LAN access**: `--lan` binds all interfaces and `--host <ipv4>` binds one address; startup
  prints the detected LAN URLs. `--web` defaults now resolve `webui/` relative to the
  executable as well as the working directory, so `GET /` no longer 404s when launched from
  `build/`.
- **Nine samplers**: `euler`, `euler_a`, `heun`, `lms`, `dpm2`, `dpm2_a`, `dpmpp_2m`,
  `dpmpp_2s_a`, `ddim`. Heun/DPM2/DPM2-ancestral use two model evaluations per step.
- **Full-checkpoint GGUF loading**: only `model.diffusion_model.*` tensors are used for the
  UNet, including native 4-D convolution dims and 1-D Q8 biases; the arena is sized from the
  UNet subset instead of the whole file. The embedded `first_stage_model.*` VAE can be read
  from the same GGUF, with LDM VAE names mapped to the diffusers layout (mid blocks,
  attention, `nin_shortcut`, `norm_out`, reversed decoder `up` blocks).
- **LoRA naming coverage**: LDM-style Kohya keys (`lora_unet_input_blocks_*`) and text-encoder
  deltas (`lora_te1_*`/`lora_te2_*`) are mapped and merged; diffusers-style keys were already
  supported.
- **Model library manifest** (`schema: sdvk.model-library/1`): declarative JSON for models,
  shared CLIP/VAE/tokenizer components, LoRAs, recommended sampler settings, sources and
  SHA256 hashes; `E:/sdvkmodel/` is the reference library and includes `fetch_loras.ps1`.

### Fixed

- `/api/unload` no longer segfaults: external arenas are destroyed before `VkDevice`, and a
  VAE-only decode no longer leaves the engine resident without releasing it.
- Euler Ancestral used a wrong coefficient and produced noise; it now follows the
  diffusers `sigma_down`/`sigma_up` formulation.
- `im2col_s2` push-constant size (20 → 24 bytes) after an `int off` was added; this was the
  cause of the earlier global "mushy image" regression.
- Web UI: job polling actually starts, 202 async responses are not treated as failures, CSS
  no longer overrides `hidden`, the device name falls back to `/api/devices`, and cancelled
  jobs are reported as cancelled rather than failed. Mobile layout no longer overflows.
- Shared libraries stored under `tokenizers/tokenizer*` are found by `resolveModels`
  (previously the UI failed with "missing tokenizer").

### Known issues

- The `Hassaku XL Lightning` Q8 GGUF currently in the local library loads but produces noise.
  Its UNet `input_blocks` weights do not match the repository's fp16 safetensors
  (correlation ≈ 0); the embedded VAE is now used, so the model file itself needs a verified
  re-export. Marked `experimental` in the manifest.
- LoRA merging reaches 722 UNet and 264 text-encoder modules without crashing, but generated
  images are still corrupted even at low strength. The merge path needs a layer-by-layer
  comparison against a reference implementation before LoRAs are used for production images.
- Performance numbers are for the RX 580 8 GB; 768² leaves little VRAM headroom and full
  checkpoint UNets may require 512–640² or smaller step/scratch arenas.

## 2026-11-05

- Fused attention (dense tiled flash kernel), vectorised q8 GEMM loads, bias epilogue fusion,
  and LoRA merge for Kohya/PEFT files. 768² sampling dropped from ~5.04 s to ~4.05–4.3 s/step.
- Direct-prompt server fixes: `/api/decode`, `/api/scan`, unloading and progress polling.

## Earlier

- SDXL pipeline (dual CLIP, UNet, VAE), GGUF Q8 support, img2img/inpaint with a real VAE
  encoder, CPU audio processing, PNG writer, Vulkan engine with arenas and command-graph
  replay. See the git history for details.
