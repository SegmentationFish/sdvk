# Changelog

All notable changes to sdvk. The project is pre-1.0; the format follows
[Keep a Changelog](https://keepachangelog.com/) and versions are approximate.

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
