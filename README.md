# sdvk

**SDXL text-to-image on a self-built Vulkan compute engine.** No deep-learning framework, no external runtime dependencies:
one CLI binary + an optional HTTP server + 70 GLSL compute shaders + [volk](src/Modules/GPU/volk) (single-file Vulkan loader)
run the whole SDXL pipeline — dual CLIP text encoding, UNet denoising and VAE decoding — directly on the GPU.

> Provenance: the inference engine is built from scratch in this project. Model weights come from third parties (see [Weights](#weights));
> this project does not include, train, or redistribute any weights.

## License

sdvk is free software, released under the **GNU General Public License v3.0** (see [LICENSE](LICENSE)).

That means: use it for anything, study and modify it, redistribute it — but if you ship a modified
version, you must ship the source under the same license. There is no warranty.
`src/Modules/GPU/volk` is [volk](https://github.com/zeux/volk) by Arseny Kapoulkine and stays under its own
MIT license; that notice is kept in `volk.h`.

## Features

- **Pure Vulkan compute**: no matrix libraries, no model framework; push descriptors, dependency-tracked minimal barriers, cached command-graph replay
- **Bring your own weights**: a model *folder* is scanned and classified by header keys, so a diffusers tree, a single-file checkpoint and a flat export all work
- **Free VAE**: any SDXL VAE (fp16 / bf16 / fp32 safetensors) can be dropped in independently of the UNet
- **One command end to end**: prompt in, PNG out (encoding → sampling → decoding → file), no intermediate tooling
- **Zero-dependency PNG**: hand-written deflate (fixed-Huffman LZ77), CRC32 and a `tEXt` block that embeds the generation parameters
- **Sampling recipes**: v-prediction and eps prediction, CFG, leading/trailing spacing, CFG rescale (mimic scale), dynamic thresholding, Euler / Euler-ancestral
- **Fused attention**: one dispatch for reshape → QKᵀ → mask → online softmax → PV → writeback; a 64×64 register-tiled flash kernel for long self-attention
- **q8 GEMM vectorised loads**: packed-word / float4 staging, adaptive 64/128-row tiles; ~2.2–3.1 TFLOP/s on RX 580
- **LoRA merge**: Kohya/PEFT safetensors, diffusers + LDM naming, conv/resnet targets, Q8 and fp32 weights
- **img2img and inpaint on a real VAE encoder**: the encoder half of the VAE is loaded and run
  (`conv_in → down blocks → mid → conv_out → quant_conv`), so a seed image becomes a latent the
  sampler understands; `--mask` blends the untouched region back every step
- **Audio workflow (host CPU)**: WAV read/write at 16/24/32-bit, peak/RMS normalise, fades, silence
  trim, Kaiser-windowed-sinc resample, FFT convolution reverb, STFT spectrogram → PNG. This is audio
  *processing*; the engine has no mel-VAE/DiT, so there is nothing to generate from a prompt
- **HTTP API + web UI**: `sdvk-server` exposes the same core, `webui/` is a dependency-free browser front end
- **Per-kernel GPU profiler**: `OFC_KERNEL_PROFILE=1` prints a GPU-millisecond table for every compute kernel

## Measured performance (RX 580 8 GB)

| Item | Value |
|---|---|
| SDXL 768×768, 20 steps, CFG 3.0 | ~4.2 s/step (sampling), ~112 s total |
| SDXL 512×512, 2 steps | ~1.95 s/step |
| q8 GEMM microbench 9216×1280×1280 | 9.65 ms (3.1 TFLOP/s) |
| VRAM usage | ~5.0 GiB in the UNet phase @768; device-local ceiling 6.0–6.1 GiB |

## Build

Requirements: CMake ≥ 3.20, a C++20 compiler, and the [Vulkan SDK](https://vulkan.lunarg.com/) (provides `glslc`).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Targets:

| Target | What it is |
|---|---|
| `sdvkcore` | static library: the SDXL pipeline with no CLI and no sockets |
| `sdvkgen` | command line tool |
| `sdvk-server` | HTTP API + static file host for `webui/` |

Turn either front end off with `-DSDVK_BUILD_CLI=OFF` / `-DSDVK_BUILD_SERVER=OFF`.

## Usage

Point it at a model folder and nothing else:

```bash
build/sdvkgen.exe -m /path/to/model "a lighthouse at dusk, oil painting"
```

A usable folder holds, in any combination:

```
model/
  noobaiXLNAIXL_vPred10Version-Q8_0.gguf    UNet (GGUF Q8_0)            <- preferred
  unet_fp16.safetensors                     UNet (diffusers or flat)   <- fallback
  text_encoder/model.safetensors            CLIP-L
  text_encoder_2/model.safetensors          OpenCLIP bigG
  vae.safetensors                           VAE (fp16 works)
  tokenizer/{vocab.json,merges.txt}         tokenizer 1
  tokenizer_2/{...}                         tokenizer 2
```

`--models <dir>` prints exactly how a folder was classified before anything loads:

```
$ build/sdvkgen.exe --models -m model
  unet:    model/noobaiXLNAIXL_vPred10Version-Q8_0.gguf
  clip-l:  model/text_encoder/model.safetensors
  clip-g:  model/text_encoder_2/model.safetensors
  vae:     model/vae.safetensors
```

Override any single component when the layout is unusual: `--unet`, `--te1`, `--te2`, `--vae`, `--tok`, `--tok2`.
A full single-file checkpoint (`.safetensors` holding `model.diffusion_model.*`, `conditioner.embedders.*`,
`first_stage_model.*`) can supply the UNet, both text encoders and the VAE at once. A full-checkpoint
**GGUF** works too: the loader keeps only the `model.diffusion_model.*` tensors for the UNet (CLIP/VAE
still come from `--te1/--te2/--vae`), sizes the arena from the UNet subset instead of the whole file,
and dequantizes 1-D Q8 biases/norms to fp32.

### Common flags

| Flag | Meaning |
|---|---|
| `--neg <text>` | negative prompt |
| `--steps <n>` / `--cfg <f>` | sampler length and guidance |
| `--size <n>` / `--width` `--height` | square edge or explicit canvas, multiples of 8 |
| `--sampler <name>` | `euler`, `euler_a`, `heun`, `lms`, `dpm2`, `dpm2_a`, `dpmpp_2m`, `dpmpp_2s_a`, `ddim` (`heun`/`dpm2`/`dpm2_a` use 2 model evals per step and run ~2x slower) |
| `--vpred` / `--zsnr` / `--dynthresh` / `--rescale <f>` | schedule and guidance options |
| `--seed <n>` | noise seed |
| `--lora <file>` / `--lora-scale <f>` | merge a LoRA safetensors at load time (default scale 1.0) |
| `--init <img.png> --denoise <f>` | img2img (the source is VAE-encoded, not pixel-seeded) |
| `--mask <mask.png> --mask-feather <n>` | inpaint; white = repaint, needs `--init` |
| `--mask-invert` | black = repaint |
| `--roundtrip` | encode then decode the seed image, no sampling; prints PSNR |
| `--out <file>` | `.png` (default) or `.ppm` |
| `--no-vae` | stop after sampling, keep the latent only |
| `--decode <latent.bin>` | VAE-only: decode a saved latent to an image |
| `-v` | per-step timing |

Recommended recipe for a v-pred checkpoint at 768 on an 8 GB card:

```bash
build/sdvkgen.exe -m model "your prompt, masterpiece" \
    --vpred --spacing leading --steps 20 --cfg 3.0 --size 768 \
    --neg "nsfw, worst quality, old, early, low quality, lowres"
```

`OFC_KERNEL_PROFILE=1` prints a per-kernel GPU-time table.

## HTTP API

```bash
build/sdvk-server.exe -m model --port 7500
```

Then open `http://127.0.0.1:7500/`. The server binds loopback by default and serves one request at a time —
a single GPU can only run one job anyway. To use the UI from a phone/tablet on the same Wi-Fi:

```bash
build/sdvk-server.exe --lan --port 7500 --web webui --out-dir output
```

It prints the LAN URLs it detects; allow TCP 7500 through the firewall if Windows asks. The HTTP API has no
authentication, so only do this on a trusted network.

| Endpoint | Purpose |
|---|---|
| `GET /` | the web UI from `webui/` |
| `GET /api/info` | version, device, loaded model |
| `GET /api/models?dir=<folder>` | classification of a model folder |
| `GET /api/scan?dir=<folder>` | per-component lists (UNet/checkpoint, VAE, CLIP-L/G, LoRA) plus defaults for the web UI |
| `POST /api/generate` | one image per call (blocking) |
| `POST /api/generate?async=1` | returns a job id immediately; poll `/api/job` |
| `POST /api/cancel?id=<id>` | stop an async job at the next sampler step |
| `GET /api/job?id=<id>` | progress, step, elapsed, output path |
| `GET /api/image?path=<file.png>` | fetch a generated or uploaded image |
| `POST /api/audio/process` | run the audio node chain over a WAV in `input/` or `output/` |
| `GET /api/audio?path=<file.wav>` | fetch a rendered WAV |
| `GET /api/devices` | Vulkan devices plus the host-RAM entry |
| `POST /api/upload` | store a PNG/WAV the UI picked, returns its path |
| `POST /api/decode` | VAE-only latent → PNG: `{"latent":"input/x.bin","vae":"...","model":"..."}` |
| `POST /api/unload` | drop the resident UNet/VAE and free device memory |

`/api/generate` also accepts `"lora": "<path>", "loraScale": 0.8` to merge a LoRA before sampling.
The LoRA is merged into the Q8 UNet at load time; `dora_scale` is currently ignored (standard LoRA is used).

```bash
curl -s -X POST http://127.0.0.1:7500/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","prompt":"a lighthouse at dusk","width":768,"height":768,
       "steps":20,"cfg":3,"vpred":true,"spacing":"trailing"}'

# img2img: init is the path /api/upload returned
curl -s -X POST http://127.0.0.1:7500/api/generate \
  -d '{"model":"model","prompt":"same scene, winter","init":"input/xyz_ref.png",
       "denoise":0.55,"width":832,"height":1216,"steps":20,"cfg":4}'

# inpaint: mask is a second upload, white = repaint
curl -s -X POST http://127.0.0.1:7500/api/generate \
  -d '{"model":"model","prompt":"red umbrella","init":"input/xyz_ref.png",
       "mask":"input/xyz_mask.png","maskFeather":8,"denoise":0.85,
       "width":832,"height":1216,"steps":20,"cfg":4}'

# audio chain: ops are applied left to right, "op:key=value,key=value" joined by ';'
curl -s -X POST http://127.0.0.1:7500/api/audio/process \
  -d '{"input":"input/voice.wav","bits":24,"spectrogram":true,
       "ops":"trim:thresholdDb=-45,padSec=0.05;normalize-peak:targetDb=-1;resample:rate=48000"}'
```

Response: `{"ok":true,"outPath":"output/sdvk_xxx.png","seed":1234,"totalMs":...,"width":768,"height":768}`.

The model stays resident between calls, so only the first request pays the load cost. Every field accepted by the
JSON body maps to a `GenParams` member; unknown keys are ignored, which keeps the API forward compatible.

## Web UI

`webui/` holds three static files (`index.html`, `style.css`, `app.js`) with no build step and no framework.
`sdvk-server` serves them from the working directory, or from `--web <dir>`:

```bash
build/sdvk-server.exe --port 7500 --spv build/shaders --web webui --out-dir output
```

Open `http://127.0.0.1:7500/` and set the model directory at the top. **扫描** walks that folder and fills the
component pickers — UNet/checkpoint, CLIP-L, CLIP-G, VAE and LoRA — while `（自动）` keeps the server-side
heuristics. Pick any file explicitly to override them, set a LoRA strength, then generate; settings and the last
prompt are kept in `localStorage`. The layout is a three-column direct-prompt studio:

- left rail: model components, LoRA (merged into Q8 at load), sampler/size, and the **Latent → PNG** card;
- centre: prompts, canvas with per-step progress, generate/cancel and download;
- right rail: output path and metrics, run log, and VRAM tips.

**清理显存** posts `/api/unload`: the UNet, CLIP and VAE weights are released and the Vulkan device is torn down, so
the next request pays the load cost again. **Latent → PNG** uploads an `.bin` latent and decodes it with the selected
VAE only — it unloads the resident UNet first so a 768×1024 decode fits in 8 GB of VRAM. `DoRA` files are currently
loaded as standard LoRA (`dora_scale` is ignored).

## Model library manifest

A model folder can describe itself with a `manifest.json` (`schema: "sdvk.model-library/1"`): for each model it
declares the UNet, CLIP-L/G, VAE, tokenizers, recommended steps/CFG/sampler, source URL and SHA256; shared
components and LoRAs get their own sections. The file is plain JSON and is meant to be edited by hand or by a
download script. `E:/sdvkmodel/` is this workstation's library, with `fetch_loras.ps1` filling the Civitai LoRAs
once `$env:CIVITAI_TOKEN` is set; sdvk itself never stores credentials.

## Weights

When publishing generated images, state: **engine = sdvk (self-built Vulkan compute); weights = third-party pretrained models**.

The weight set used by this project (not bundled; obtain them yourself):

- **NoobAI-XL vpred-v1.0** (SDXL architecture, v-prediction) — official Q8_0 GGUF conversion from [civitai](https://civitai.com/models/833294)
- **SDXL VAE** (madebyollin/sdxl-vae-fp16-fix or equivalent) and the **CLIP-L / OpenCLIP bigG** text encoders — Stability AI open components

## Layout

```
src/Modules/
  Diffusion/  sdvkgen.cpp (pipeline), Eval.hpp (library API), GenParams, ModelSet (folder classification),
              main.cpp (CLI), Safetensors reader
  Server/     sdvk-server: HTTP + JSON, no dependencies
  GPU/        Vulkan engine (arenas, barrier tracking, kernel table, profiler) + 70 compute shaders + volk
  Model/      GGUF reader (Q8_0)
  Core/       tensor buffer helpers
webui/        static web front end
```

## Roadmap

- [x] Direct PNG output (built-in, zero-dependency)
- [x] `--decode` mode: load the VAE only and decode a latent
- [x] Free model / VAE selection
- [x] HTTP API and web UI
- [ ] In-engine VAE decode at 768 (step-arena budget / chunked decode)
- [ ] `--ckpt`: single-file fp checkpoint loading (on-the-fly rank-2 quantization to Q8_0)
- [x] Hotspot optimization: fused tiled attention, vectorised q8 GEMM loads, bias epilogue
- [x] LoRA merge (Kohya / PEFT, conv + attention targets)
- [ ] DoRA magnitude application
- [ ] int4 weight variant / Winograd 3×3

## Tools

`tools/costmodel.py` is a GPU-free cost model: it parses the GGUF header, recovers the real conv
shapes from `comfy.gguf.orig_shape.*` and ranks every node by estimated compute or bandwidth time.

```bash
python tools/costmodel.py model/noobaiXLNAIXL_vPred10Version-Q8_0.gguf --latent 96
```

```
total           4172.4 GFLOP    19.76 GB
bound         compute
floor             0.76 s/pass  (rx 580 measured 14.1 s/pass)
```

That is the mathematical floor for one UNet pass; the gap to 14.1 s is where the optimization work is.

`tools/smoke.sh` rebuilds both targets and exercises every path that does not need a GPU: the CLI
help/version/usage routes, the cost model, and the HTTP endpoints including the path-traversal guard.

`tools/stress.sh` is the GPU matrix for img2img / inpaint (roundtrip PSNR, denoise ladder, mask
feathers, 20-run step-time drift). It launches real generations, so run it yourself and watch the
desktop — a device-local request the driver cannot satisfy takes the whole session down:

```bash
tools/stress.sh -m E:/ofc-data/noobai -i output/yae_miko.png -M input/mask.png
```

`build/audio_test.exe` is the GPU-free audio check: PCM roundtrips, resample distortion, FFT vs
direct convolution, fade/normalise/trim values, spectrogram peak. Every line prints PASS/FAIL and
the exit code is non-zero on a failure.


## Weight licensing

sdvk's own code is GPL-3.0-only, but that says nothing about the weights you generate with. The
checkpoints the project targets carry their own terms — **NoobAI-XL is under the Fair AI Public
License (faipl-1.0-sd)**, the SDXL VAE and the CLIP text encoders come from Stability AI's
open weights, and any LoRA you load may differ again.

Practical consequences:

- The engine's GPL license does **not** grant rights to the weights or to images you produce with them.
- Read the model card before commercial use; the Fair AI licenses restrict redistribution and require
  you to pass the license on with the model.
- The `tEXt` block frozen into every PNG names the checkpoint, so anyone can check which weights a
  picture came from.
