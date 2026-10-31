# sdvk 交接文档（2026-10-25 更新）

> 给下一个 AI 会话/工具的完整上下文。读完即可继续干活，无需其他背景。

## 0. 项目一句话

sdvk = 自研纯 Vulkan compute 的 SDXL 文生图引擎（C++/GLSL，无第三方 runtime），已发布 GitHub：`https://github.com/SegmentationFish/sdvk`（main 分支，公开）。

**权重归属声明（必须遵守）**：引擎代码自研；生成权重为 NoobAI-XL vpred-v1.0（civitai 官方 Q8_0 GGUF）+ SDXL VAE + CLIP-L / OpenCLIP bigG（F16 safetensors），均不随代码分发。任何演示/成果汇报必须标注"引擎归属 + 权重来源"。

## 1. 环境与硬约束（红线，永久有效）

- 用户：学生（升高中），零成本优先，速度优先、只认实测数字。
- 硬件：RX 580 8GB（device-local 可用 ~6 GiB），16GB RAM（常剩 3–4GB，浏览器很吃 VRAM/RAM）。
- **GPU 严格串行**：跑 GPU 前先 `tasklist | grep -iE "sktrain|sdvkgen|sdvk-server"` 确认无占用；遗留的 sdvk-server 会占 GPU 和端口（实测踩过：残留进程让新 server 请求全部打到旧进程）。
- **平台通用性红线**：不打平台专属补丁，修复必须对真机/其他环境同样正确，优先标准接口。
- **长任务用户自己跑**；**交付优先**：改好代码交用户实测、最小里程碑推进。
- 开发：Windows + MSYS2 + CLion（MinGW g++）；bash cwd 会漂移，每条命令显式 `cd` 绝对路径。
- 构建：CMake + Ninja，`cd build && "D:/JETBRA~2/CLion/bin/ninja/win/x64/ninja.exe" sdvkgen sdvk-server`。
- **运行目录**：shader 默认相对路径 `shaders`，要么 `cd build` 运行，要么显式 `--spv build/shaders`。

## 2. 代码架构（2026-10-24 重构后）

- `src/Modules/Diffusion/sdvkgen.cpp` → **sdvkcore 静态库**：全部推理管线（TE/UNet/采样器/VAE/PNG/OFCL 读写/内核封装）。对外接口在 `Eval.hpp`：`run()`、`decodeLatentFile()`。
- `src/Modules/Diffusion/main.cpp` → **CLI `sdvkgen`**：argv → `GenParams`，打印人类可读输出。
- `src/Modules/Diffusion/GenParams.{hpp,cpp}` → 参数结构（CLI / HTTP / WebUI 共用）。
- `src/Modules/Diffusion/ModelSet.{hpp,cpp}` → **自由模型加载**：`-m <dir>` 自动发现 unet/TE1/TE2/VAE/tokenizer；`--unet/--te1/--te2/--vae/--tok/--tok2/--taesd` 组件级覆盖；`--models` 打印分类结果。
- `src/Modules/Server/server.cpp` → **sdvk-server**：HTTP API + 托管 `webui/`。
- 其他目标：`vkbudget`（显存预算）、`ggufinfo`（GGUF 元数据）。
- CLI 能力（`--help` 全量）：采样（steps/cfg/seed/sampler euler|euler_a/spacing/vpred/zsnr/rescale/dynthresh/eta）、画布（--size / --width/--height / --init/--crop img2img）、输出（--out .png|.ppm、--png-compress 0|1、--no-vae、--latent-out/--no-latent、--no-info）、运行时（--spv/--param-mb/--step-mb/--scratch-mb/--vstep-mb/-v）。
- 二进制在 `build/`：`sdvkgen.exe`、`sdvk-server.exe`（均已按本轮修复重新构建）。

## 3. 本轮验证结果（2026-10-25，RX 580 实测）

**已通过**：
- `--help` / `--version` / `--models -m E:/ofc-data/noobai`（自动发现正确，带 note）。
- `--decode` 768 latent（96×96 OFCL）→ PNG：VAE 4.9–5.4s，默认 step arena 2400MB 可跑。
- 768 全流程 8 步：**13.6 s/step**，含 VAE 总 ~127s，输出 PNG；2 步 ~63s。
- **768×1024 全流程 30 步**：18.3 s/step（总 ~9.5 min），`--size 768x1024` 现已支持（WxH 解析）；latent 正确写出。该尺寸解码需 ~3.3GB step arena，默认 2400 不够：用 `--vstep-mb 3600`。
- **VAE 解码保真度**：用 `tools/refdecode.py`（torch CPU 参考解码）解同一 latent 对比，构图/曝光一致，但逐像素 mean|diff| ≈ 31/255（引擎细节更糊）→ 解码器仍有残余偏差（怀疑 up2/attention，待逐层对拍）。
- **画质现象**：30 步 + cfg 4.0 出图严重过曝/绘画感；参考解码同样如此 → 是采样 latent 本身，不是写图/解码 bug。已验证配方是 **cfg 3.0**（README），cfg 4.0 偏热；vpred 可试 `--rescale 0.7` 压过曝。
- PNG 字节级校验（Python 手解：签名/各 chunk CRC/IHDR/zlib/adler32/filter/逐像素）：全过；tEXt 参数块正确（脚本 `/d/tmp/pngtest/verify_any.py`）。
- server：`GET /`（webui）、`/api/info`、`/api/models`、非法请求 400 正常。
- 证据图：`D:/tmp/pngtest/gen768_8s.png`（8 步，能看出向日葵田结构；粗糙是步数限制）、`const0.png`（常数 latent → 均匀灰，解码器结构正确的判据）。

**本轮修复的真 bug（7 个，均在 sdvkgen.cpp / vk_engine.cpp）**：
1. `decodeLatentFile` 不调用 `resolveModels` → `--decode -m` 被忽略、VAE 路径为空。
2. decode 的 info 条件写反（`p.infoOut ? "" : infoText`）。
3. PNG 压缩路径把完整 deflate 流又包了一层 stored block → 所有压缩 PNG 无效。
4. fixed-Huffman 长度码映射错（280–285 的码字是 192+(li−23)，不是 257+li）→ 位流错乱。
5. `vaeResnet` 的 `flip()` 调用顺序使 src/dst 同名 → 3×3 conv 原地读写、数据被自己覆盖（改为 A/B 乒乓）。
6. `vaeAttn` 在 `linear()` 前 `arenaRewind(o.offset)` → 投影输出写回 gemm 输入（删除该 rewind）。
7. **`compact()`/`Arena::moveDown` 在命令录制期提交独立 command buffer 并等队列空闲**——此时本步 dispatch 还只在录制中没提交，搬的是过期内存、还覆盖了尚未被读的输入。这是"解码输出与输入无关的彩虹条纹"的根因。修法：新增 `Arena::moveDownInCmd()`（把 bounce 拷贝录进当前 stepCmd_）与 `Engine::compactStepArena()`，`compact()` 改用它。
8. **`spatialTransformer` 的 `arenaRewind(o.offset)`**：transpose2 之后把 arena 回退到 o 的起点，`conv1(o, proj_out)` 的输出与输入同址，fp32 gemm 原地读写 → UNet 每个 transformer 块的注意力输出被破坏（改为 `arenaRewind(o.offset + o.size)`）。

**画质状态（2026-10-25 已验收）**：阿米娅 768×1024 / 30 步 / cfg 3.0 的 latent 用 `tools/refdecode.py` 参考解码得到干净动漫图；此前"碎玻璃"高频噪声的根因是第 9 个 bug：
9. **`vaeAttn` 布局错位**：新版用 `linear()` 直接处理通道优先的卷积张量 `[C,P]`（旧版用 `conv1`），且 attnCore 输出 token 优先却直接与 `x` 相加。修复：进出各 `transpose2` 一次。
修复后引擎解码 vs torch 参考：图像干净。随后按用户反馈继续查"颜色偏/噪点"，又修掉：
10. **`tools/refdecode.py` 自身 3 处 bug**：激活函数写成 erf-GELU（SDXL VAE 是 SiLU）、to_q/to_k/to_v 漏加 bias；修复后阶段级对拍与引擎一致到 ≤0.015（常数与随机 latent 均是）。
11. **`Arena::moveDown` 的 c1→c2 之间缺 transfer→transfer 屏障**——Vulkan 不保证连续 copy 的写对后一条 copy 可见，compact 会搬出过期数据（已加屏障）。
12. 调试开关：`OFC_VAE_TRACE=1`（阶段 dump，仅小图）、`OFC_VAE_TRACE_SUB=1` + `OFC_VAE_TRACE_RESNET=<子串>`（指定 resnet 子步）、`OFC_NO_COMPACT=1`。
13. **`vaeDecode` 的 up_blocks 循环没有在第一个 resnet 后把 `ci = co`**：up2（512→256）与 up3（256→128）的 resnets.1/2 用块入口宽度去归一化/卷积实际已变窄的张量（GN 统计与 conv 读错通道），表现为从 `up_blocks.2.resnets.1` 开始幅度暴涨、最终颜色偏浓/发暗。up0/up1 因 ci==co 未暴露。修复后引擎解码 vs 修正后参考：**mean|diff| 0.46–0.50/255，max 1（纯 fp32 舍入）**，随机 latent 与 768×1024 实图均一致。
**画质状态：VAE 解码器已与参考逐像素对齐（≤1/255）。** 产物：`output/amiya_v3_engine_fixed3.png`（引擎自解码，30 步 cfg3.0）。
**自动内存（2026-10-31）**：① UNet step arena 启动时用 plan walk（纯 host 计数，复用 OFC_ARENA_PLAN 的计数器）按画布自动定容 = 峰值×1.05+48MB，`--step-mb` 只当下限（768×1024 自动算出 1899MB，替代此前 1900 差 20MB 就整轮失败的痛点）；② VAE 解码改为 `400 + H*W*290/1024 MB`（封顶 4000），96×128 自动 3880MB（旧公式 2400 上限会 OOM）；`OFC_NO_AUTOSIZE=1` 可关闭两者、完全按用户值。
**路线说明**：运行时/引擎保持纯 Vulkan + C++（无 torch/diffusers 依赖）；`tools/ref*.py` 仅作离线数值验证参考，不进入产品链路。
`tools/refunet_probe.py` 仍有脚本 bug（GEGLU 公式、linear 形状崩溃），UNet 逐层对拍待修。
**画质说明**：8 步 vpred 只能出粗结构；验收用 28~30 步（cfg 3.0、trailing、负向提示）。

## 4. 性能现状与剖析

- 实测（2026-10-25 优化后）：**768² 4.85 s/step**（2 步基准，此前 13.8）；768×1024 30 步预计 ~3.5 min。VAE 768 解码 ~5–8.5s。
- **已完成优化**：① GroupNorm+SiLU 融合（groupnorm 加 act 标志，push 16→20B；UNet resBlock×2 / out 链、VAE resnet×2 / conv_norm_out 合并，逐像素 0/255 差异）；② **GEMM 共享内存 bank 冲突修复**（As/Bs 加 +1 列 padding，步长与 32 互质）：gemm_nt_q8 21.1s→5.1s（**4.1×**），gemm_nt 1.87s→0.64s，总 GPU 时间 28.3s→10.1s。**BK=16→32 已验证为负优化（6.01 s/step），已回退**。
- 剖析（768²/2 步）：gemm_nt_q8 51.7%、gemm_nn_q8 21.8%、gemm_nt 5.9%、gemm_nn 5.2%、softmax 3.2%、copy_nd 2.7%、ew2 2.5%、im2col 1.5%。注意力全链路 ~14%，GEMM 仍是主战场。
- 下一步优化候选：gemm_nn_q8（BN=128 减 A 权重复读/去量化；或 conv 权重一次性反量化到 fp16 缓冲）、QKV 三合一 GEMM、softmax 融入 AV GEMM（flash 式，注意力 ~14%）、bias 尾融合（bias_add 0.4%）。
- `tools/costmodel.py` 已修好（读 `comfy.gguf.orig_shape.*`）；96² 理论下限 0.76 s/pass 的模型估计与实测差距需按新数据复核。
- **剖析器**：`OFC_KERNEL_PROFILE=1` 每次运行末尾打印内核表（已实测可用）。

## 5. 已知问题 / 下一步（按优先级）

1. **sdvk-server `/api/generate` 会崩**（高优先级）：静态页、/api/info、/api/models 正常；POST 生成请求后进程在模型加载完成后直接退出（日志 `/tmp/srv2.log` 最后停在 `device: AMD Radeon RX 580 2048SP`）。CLI 同一管线正常，怀疑 worker 线程（`std::thread` 跑 run()）或进度回调/引擎线程亲和问题。次生：从 `build/` 启动要传**绝对路径** `--web <repo>/webui`（相对 `../webui` 被 safeJoin 拒 → 404）。
2. **28 步画质验收**：`cd build && ./sdvkgen.exe -m E:/ofc-data/noobai "sunflower field, masterpiece, best quality" --vpred --spacing trailing --steps 28 --cfg 3.0 --size 768 --seed 42 --neg "nsfw, worst quality, old, early, low quality, lowres" --scratch-mb 700 --out out.png`（约 6.5 min）。
3. **VRAM 余量**：768 时 step 1500 + scratch 900 在浏览器占用下会在 step 3 报 VK -2；用 `--scratch-mb 700` 可过。非方形 768×1152 未复测（yae_miko 场景）。
4. **未验证分支**：`--unet` 单文件完整 checkpoint（#50）、`--taesd`、img2img `--init/--crop`、euler_a、zsnr。
5. **发布**：工作区 ~82 文件改动（含本轮修复）未提交；README 可能仍描述旧 CLI。发布前 review → commit → push。**不要提交凭据**。
6. 旧任务 #48（PNG 直出）、#49（--decode/768 解码）本轮已实测，可关单。

## 6. 关键技术事实（快速上手）

- **采样推荐配方**：`--vpred --spacing trailing --steps 28 --cfg 3.0 --size 768 --neg "nsfw, worst quality, old, early, low quality, lowres"`。512 是 SDXL 训练分布外（OOD），效果差。
- **CFG 机制**：`resv = epsU + cfg·(epsC − epsU)`；词从负向挪到正向 = 差向量贡献符号翻转。
- **GGUF conv 形状**：存 2D 压缩 dims，真实形状在 `comfy.gguf.orig_shape.<name>`（rows=u64s[0]，cols=其余乘积）；`ggdims()` 优先读它。
- **OFCL latent**：magic "OFCL" + `3×u32 {4,H,W}` + fp32（采样器空间，解码时 `/0.13025`）。旧版本写过 `{4,h,h}` 的坏头，文件可能比头声明大——以文件大小交叉验证。
- **VAE 解码显存**：自动 `vStepMB = max(--vstep-mb, 400 + H*W*290/1024 MB)`（封顶 4000；96×128 → 3880MB）；UNet step arena 由启动时的 plan walk 自动定容（峰值×1.05+48MB，`--step-mb` 为下限，`OFC_NO_AUTOSIZE=1` 关闭）。`vaeResnet` 槽位复用 + `compact()` 把活跃张量压到基址 0（必须是**录制进命令流**的拷贝，且 bounce 的两段 copy 之间要 transfer→transfer 屏障）。
- **引擎命令流**：dispatch/拷贝全录进 `stepCmd_`，`endStep()` 才提交；`inStep_` 期间严禁另开 command buffer 并 `vkQueueWaitIdle`。
- **内核**：`gemm_nt_q8.comp`（16×16 wg，BM=BN=64，BK=16，TM=TN=4 寄存器块，deqB 内联反量化：uint32 打包 int8 + fp32 scale 每 32 元素）；conv 走行分块 im2col + gemmNnQ8（kColFloats=12M）；VAE resnet 槽位复用。
- **错误码**：VK -1 = host OOM（RAM 不足）；VK -2 = device OOM。arena OOM 会打印缺口 MB。
- **配置**：CLI 默认 `steps=20, cfg=7.5, size=512`；模型路径无默认值，必须 `-m` 或显式组件路径。
- **repo**：`src/Modules/GPU/`（vk_engine + shaders/*.comp）、`src/Modules/Diffusion/`、`src/Modules/Server/`、`webui/`、`tools/`；`.clang-format`（Google 基 + ColumnLimit 120）。

## 7. 产品路线（2026-10-31，与用户确认的方向）

- **NodesVkimagine（UI）**：ComfyUI 前端已提取 → `D:/tmp/comfyui/web/comfyui_frontend_package/static`（wheel `comfyui-frontend-package==1.57.6`，98MB，index.html+assets；未改其源码，GPL-3.0 与本仓库兼容，分发时保留其 LICENSE/署名）。前端依赖端点：`/object_info`、`/prompt`、`/view`、`/ws`、`/system_stats`（另有 /queue、/history、/interrupt 等）。下一步：在 `server.cpp` 实现这组协议 + 图执行器，最小节点集 `CheckpointLoaderSimple / CLIPTextEncode / KSampler / VAEDecode / SaveImage` → `run()`；之后可做 desktop 包装（Tauri/WebView2，内存与浏览器标签页相当，不省）。
- **兼容性**：下一步做 **SD1.5** 适配（单 CLIP-L、无 CLIP-G、VAE scale 0.18215、调度差异）+ ModelSet 架构识别/适配器接口。FLUX/SD3/视频模型是换引擎的量级，且 6GB 显存放不下量化权重（FLUX Q4 仍 >7GB）——需换卡后再启动。
- **提速**：少步蒸馏是正解（LCM/Lightning/Turbo，4–8 步、cfg 1–2）→ 单图约 10–20s（当前 30 步 × 4.9s ≈ 147s）；引擎侧只需加 LCM 调度 + cfg≤1 跳过 uncond pass。需要权重方案：LoRA（需加权重合并）或 turbo GGUF（需找/量化）——待用户定。
- 辅助优化余量：gemm_nt_q8 约占 52%（已试 BN=128 负优化回退；下一步 QKV 合一/向量化加载 ~1.3×）、注意力融合 ~3–5%、copy_nd/ew2 再融合。

## 8. 安全备注

- 用户曾发过一个 GitHub PAT 用于建仓推送，**已建议用后撤销**；remote URL 中 token 已擦除（`https://github.com/SegmentationFish/sdvk.git`）。今后不索取凭据、不把凭据写入任何文件或提交。
