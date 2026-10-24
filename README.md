# sdvk

自研 Vulkan compute 推理引擎上的 **SDXL 文生图**。无深度学习框架、无外部运行时依赖：
一个 `sdvkgen.exe` + 66 个 GLSL compute shader + [volk](src/Modules/GPU/volk)（单文件 Vulkan 加载器），直接在 GPU 上完成 SDXL 的双 CLIP 文本编码与 UNet 去噪。

> 归属声明：推理引擎为本项目自研；模型权重来自第三方（见下方 [权重来源](#权重来源)），本项目不包含、不训练、不重新分发任何权重。

## 特性

- **纯 Vulkan compute**：无矩阵库、无模型框架；push descriptors、依赖跟踪的最小屏障、命令图重放缓存
- **Q8_0 GGUF UNet**：SDXL UNet 以 Q8_0 量化 GGUF 加载，GEMM shader 内现场反量化（34 字节/32 权重块布局）
- **双 CLIP 文本编码器**：CLIP-L + OpenCLIP bigG，token 截断/BOS/EOS 与 SDXL 语义一致（ pooled 输出齐备）
- **采样配方**：v-prediction 与 eps 两种预测、CFG、DDIM trailing spacing、CFG rescale（mimic scale）、dynamic thresholding
- **显存适配**：param / step / scratch 三 arena 全局规划，面向 8GB 消费卡（实测 6.0 GiB device-local 可用）
- **每内核 GPU 剖析器**：`OFC_KERNEL_PROFILE=1` 输出全部 compute kernel 的 GPU 毫秒分布表

## 实测（RX 580 8GB）

| 项目 | 数值 |
|---|---|
| SDXL UNet 前向 @768×768（latent 96²） | 14.1 s / pass（CFG 一步 = 2 pass） |
| SDXL UNet 前向 @512×512 | 6.9 s / pass |
| 显存占用 | UNet 阶段 ~5.0 GiB（768）；device-local 天花板 6.0–6.1 GiB |
| 768×768 / 28 步 / CFG 3.0 出图（UNet 部分） | ~6.5 分钟 |

## 构建

依赖：CMake ≥ 3.20、C++20 编译器、[Vulkan SDK](https://vulkan.lunarg.com/)（提供 `glslc`）。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sdvkgen
```

## 使用

需要四个权重文件（路径可用参数覆盖）：

| 文件 | 默认路径 | 说明 |
|---|---|---|
| UNet GGUF (Q8_0) | `noobaiXLNAIXL_vPred10Version-Q8_0.gguf` | ComfyUI 键名的 SDXL UNet 量化版 |
| CLIP-L | `text_encoder__model.safetensors` | diffusers 键名 |
| OpenCLIP bigG | `text_encoder_2__model.safetensors` | OpenCLIP 原生键名 |
| VAE | `vae.safetensors` | ldm 键名（fp16/fp32/bf16 均可读） |

v-pred 模型的推荐配方（768）：

```bash
build/sdvkgen.exe "your prompt, masterpiece" \
    --vpred --spacing trailing --steps 28 --cfg 3.0 --size 768 \
    --no-vae --step-mb 1500 --scratch-mb 500 \
    --neg "nsfw, worst quality, old, early, low quality, lowres"
```

- `--no-vae`：跳过引擎内 VAE 解码，输出 `sdvkgen_latent.bin`（显存紧张设备的推荐路径；8GB 卡上引擎内 768 解码当前超出显存预算，优化中）
- latent 可用任何 SDXL VAE 解码脚本出图（如 torch 实现，注意 ×0.13025 scaling factor）
- `OFC_KERNEL_PROFILE=1 ./sdvkgen.exe ...` 输出每内核 GPU 耗时表（stderr）
- `--textcheck`：导出文本编码器中间结果用于对拍

## 权重来源

演示/发布出图时请标注：**引擎 = 本项目自研 sdvk（Vulkan compute）；权重 = 第三方预训练模型**。

本项目使用的权重组合（均未随代码分发，需自行获取）：

- **NoobAI-XL vpred-v1.0**（SDXL 架构，v-prediction）— [civitai](https://civitai.com/models/833294) 官方提供的 Q8_0 GGUF 转换
- **SDXL VAE**（madebyollin/sdxl-vae-fp16-fix 或等效）与 **CLIP-L / OpenCLIP bigG** 文本编码器 — Stability AI 开源组件

## 目录结构

```
src/Modules/
  GPU/        Vulkan 引擎（arena、屏障跟踪、kernel 表、剖析器）+ 66 个 compute shader + volk
  Diffusion/  sdvkgen：SDXL 管线（CLIP 编码 / UNet 去噪 / VAE 解码）、safetensors 读取器
  Model/      GGUF 读取器（Q8_0）
  Core/       张量缓冲辅助
```

## 路线

- [ ] PNG 直出（内置 zlib 级压缩，零依赖）
- [ ] 引擎内 VAE 解码 768 跑通（step arena 预算 / 分块解码）
- [ ] `--decode` 模式：只载 VAE 解码 latent
- [ ] `--ckpt`：单文件 fp checkpoint 加载（现场 rank-2 量化为 Q8_0）
- [ ] 热点优化（每内核剖析数据驱动：GEMM tile / 融合 / 屏障开销）
