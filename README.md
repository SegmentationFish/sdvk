# sdvk

**SDXL text-to-image on a self-built Vulkan compute engine.** No deep-learning framework, no external runtime dependencies:
one `sdvkgen.exe` + 66 GLSL compute shaders + [volk](src/Modules/GPU/volk) (single-file Vulkan loader) run the full SDXL pipeline —
dual CLIP text encoding and UNet denoising — directly on the GPU.

> Provenance: the inference engine is built from scratch in this project. Model weights come from third parties (see [Weights](#weights));
> this project does not include, train, or redistribute any weights.

## Features

- **Pure Vulkan compute**: no matrix libraries, no model framework; push descriptors, dependency-tracked minimal barriers, cached command-graph replay
- **Q8_0 GGUF UNet**: the SDXL UNet loads as Q8_0-quantized GGUF, dequantized inside the GEMM shaders (34-byte / 32-weight block layout)
- **Dual CLIP text encoders**: CLIP-L + OpenCLIP bigG with SDXL-faithful token truncation / BOS / EOS semantics (pooled outputs included)
- **Sampling recipes**: v-prediction and eps prediction, CFG, DDIM trailing spacing, CFG rescale (mimic scale), dynamic thresholding
- **Memory planning**: param / step / scratch arenas sized globally, targeting 8 GB consumer cards (6.0 GiB device-local usable in practice)
- **Per-kernel GPU profiler**: `OFC_KERNEL_PROFILE=1` prints a GPU-millisecond table for every compute kernel

## Measured performance (RX 580 8 GB)

| Item | Value |
|---|---|
| SDXL UNet forward @768×768 (latent 96²) | 14.1 s / pass (one CFG step = 2 passes) |
| SDXL UNet forward @512×512 | 6.9 s / pass |
| VRAM usage | ~5.0 GiB in the UNet phase @768; device-local ceiling 6.0–6.1 GiB |
| 768×768 / 28 steps / CFG 3.0 (UNet only) | ~6.5 min |

## Build

Requirements: CMake ≥ 3.20, a C++20 compiler, and the [Vulkan SDK](https://vulkan.lunarg.com/) (provides `glslc`).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sdvkgen
```

## Usage

Four weight files are required (paths overridable via CLI flags):

| File | Default path | Notes |
|---|---|---|
| UNet GGUF (Q8_0) | `noobaiXLNAIXL_vPred10Version-Q8_0.gguf` | SDXL UNet with ComfyUI key names |
| CLIP-L | `text_encoder__model.safetensors` | diffusers key names |
| OpenCLIP bigG | `text_encoder_2__model.safetensors` | native OpenCLIP key names |
| VAE | `vae.safetensors` | ldm key names (fp16 / fp32 / bf16 all readable) |

Recommended recipe for a v-pred checkpoint at 768:

```bash
build/sdvkgen.exe "your prompt, masterpiece" \
    --vpred --spacing trailing --steps 28 --cfg 3.0 --size 768 \
    --no-vae --step-mb 1500 --scratch-mb 500 \
    --neg "nsfw, worst quality, old, early, low quality, lowres"
```

- `--no-vae` skips the in-engine VAE decode and writes `sdvkgen_latent.bin` (recommended on 8 GB cards; the in-engine 768 decode currently exceeds the VRAM budget and is being optimized)
- decode the latent with any SDXL VAE script (e.g. a torch implementation; mind the ×0.13025 scaling factor)
- `OFC_KERNEL_PROFILE=1 ./sdvkgen.exe ...` prints a per-kernel GPU-time table (stderr)
- `--textcheck` dumps text-encoder intermediates for parity checks

## Weights

When publishing generated images, state: **engine = sdvk (self-built Vulkan compute); weights = third-party pretrained models**.

The weight set used by this project (not bundled; obtain them yourself):

- **NoobAI-XL vpred-v1.0** (SDXL architecture, v-prediction) — official Q8_0 GGUF conversion from [civitai](https://civitai.com/models/833294)
- **SDXL VAE** (madebyollin/sdxl-vae-fp16-fix or equivalent) and the **CLIP-L / OpenCLIP bigG** text encoders — Stability AI open components

## Layout

```
src/Modules/
  GPU/        Vulkan engine (arenas, barrier tracking, kernel table, profiler) + 66 compute shaders + volk
  Diffusion/  sdvkgen: the SDXL pipeline (CLIP encoding / UNet denoise / VAE decode), safetensors reader
  Model/      GGUF reader (Q8_0)
  Core/       tensor buffer helpers
```

## Roadmap

- [ ] Direct PNG output (built-in, zero-dependency)
- [ ] In-engine VAE decode at 768 (step-arena budget / chunked decode)
- [ ] `--decode` mode: load the VAE only and decode a latent
- [ ] `--ckpt`: single-file fp checkpoint loading (on-the-fly rank-2 quantization to Q8_0)
- [ ] Hotspot optimization (driven by the per-kernel profiler: GEMM tiles / fusion / barrier overhead)
