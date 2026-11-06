#include "Modules/GPU/fused_attn.hpp"
// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// SDXL (NoobAI-XL vPred-1.0, Q8_0 GGUF) text-to-image on the ofc Vulkan engine.
// fp32 activations; Q8_0 weights stay as int8 payload + fp32 scale in VRAM and are
// dequantized inside the gemm shaders (gemm_nt_q8 / gemm_nn_q8).
//
// Pipeline: CLIP-L (768, 12 layers) + OpenCLIP-bigG (1280, 32 layers, pooled) ->
// UNet2DConditionModel (SDXL: added_time_embed 2816, cross-attn 2048, heads 5/10/20) ->
// EulerDiscreteScheduler (epsilon) -> VAE decoder (fp16 weights, dequantized on load).
//
// Weight phases share one param arena (reset between phases) to stay inside 7.75 GiB VRAM:
//   A) text encoders   B) UNet   C) VAE
#include "Modules/GPU/gpu_ops.hpp"
#include "Modules/Diffusion/Safetensors.hpp"
#include "Modules/Model/Gguf.hpp"
#include "GenParams.hpp"
#include "ModelSet.hpp"
#include "Eval.hpp"

using ofc::sdxl::EvalModel;

using ofc::sdxl::Progress;
using ofc::sdxl::ProgressFn;
using ofc::sdxl::RunResult;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ofc;
using namespace ofc::gpu;
using ofc::sdxl::GenParams;
using ofc::sdxl::ModelPaths;
using ofc::sdxl::ModelSet;
using ofc::sdxl::SamplerKind;

// ================================================================ globals

// the model currently resident in the parameter arena (one loader serves every component)
struct ModelFile {
    std::string path;
    int kind = 0;  // 0 none, 1 gguf (UNet), 2 safetensors
    GgufFile gg;
    Safetensors st;
};
static ModelFile gUNet, gTe1, gTe2, gVae;
static int gArch = 2;  // 1 = SD1.5 (LDM UNet, single CLIP-L), 2 = SDXL
static std::string gDeviceName;

static std::vector<std::pair<std::string, Safetensors*>> gSTs;  // (prefix, file) for dim lookup

static std::unordered_map<std::string, BufRef> gW;   // fp32 weights
static std::unordered_map<std::string, BufRef> gWQ;  // Q8_0 int8 payloads
static std::unordered_map<std::string, BufRef> gWS;  // Q8_0 fp32 scales
// logical (rows, cols) of every fp32 weight we uploaded. Checkpoint slices rename tensors on the
// way in, so the file-based findST lookup cannot resolve them; this table keeps the dims.
static std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> gDims;

static uint32_t ceilDiv(uint64_t a, uint32_t b) { return (uint32_t)((a + b - 1) / b); }

// Step-by-step arena demand probe: OFC_ARENA_PLAN=1 prints the peak the UNet pass needs before
// anything is allocated, so a canvas can be sized without trial-and-error OOMs.
static bool gArenaPlan = false;
static bool gAutoPlan = false;  // planning walk used to auto-size the arenas; it must not exit
static uint64_t gPlanUsed = 0, gPlanCap = 1ull << 62, gPlanPeak = 0;
static void planNote(uint64_t usedBytes) {
    if (usedBytes > gPlanPeak) gPlanPeak = usedBytes;
}
static BufRef allocN(uint64_t nFloats) {
    if (gArenaPlan) {
        uint64_t off = (gPlanUsed + kAlign - 1) / kAlign * kAlign;
        gPlanUsed = off + nFloats * 4;
        planNote(gPlanUsed);
        return BufRef{VK_NULL_HANDLE, off, nFloats * 4};
    }
    return G.eng.stepArena().alloc(nFloats * 4);
}
static void planReset() {
    gPlanUsed = 0;
    gPlanPeak = 0;
}
// In plan mode nothing is allocated and nothing may reach the GPU: a fake BufRef has no handle,
// so dispatching it would block forever. These two wrappers keep the pass a pure counter walk.
static void gDisp(const char* kernel, const std::vector<BufRef>& bufs, const void* push, uint32_t gx,
                  uint32_t gy = 1, uint32_t gz = 1) {
    if (gArenaPlan) return;
    G.eng.dispatch(kernel, bufs, push, gx, gy, gz);
}
static void gBar() {
    if (gArenaPlan) return;
    G.eng.barrier();
}
static uint64_t arenaMark() {
    if (gArenaPlan) return gPlanUsed;
    return G.eng.stepArena().used();
}
static void arenaRewind(uint64_t bytes) {
    if (gArenaPlan) {
        if (bytes < gPlanUsed) gPlanUsed = bytes;
        return;
    }
    G.eng.stepArena().rewind(bytes);
}

static void die(const std::string& m) {
    std::printf("FAIL: %s\n", m.c_str());
    std::exit(1);
}

// one error type for everything the caller can act on; run() turns it into an error line
static void fail(const std::string& m) { throw std::runtime_error(m); }

// ================================================================ weight access

static const BufRef& W(const std::string& name) {
    auto it = gW.find(name);
    if (it == gW.end()) fail("missing fp32 weight " + name);
    return it->second;
}
static bool hasW(const std::string& name) { return gW.count(name) != 0; }

static const BufRef& WQ(const std::string& name) {
    auto it = gWQ.find(name);
    if (it == gWQ.end()) fail("missing Q8 payload " + name);
    return it->second;
}
static const BufRef& WS(const std::string& name) {
    auto it = gWS.find(name);
    if (it == gWS.end()) fail("missing Q8 scale " + name);
    return it->second;
}
static bool isQ8(const std::string& name) { return gWQ.count(name) != 0; }
// a weight may live in the fp32 map or the Q8_0 payload maps
static bool hasAnyW(const std::string& name) { return gW.count(name) || gWQ.count(name); }

// GGUF logical (rows, cols): comfy.gguf.orig_shape.* wins, else ne[1]xne[0] (1D -> ne0 x 1, like sdims)
static void ggdims(const std::string& name, uint32_t& rows, uint32_t& cols) {
    const GgufTensor* t = gUNet.gg.find(name);
    if (!t) fail("missing gguf tensor " + name);
    const GgufMeta* m = gUNet.gg.meta("comfy.gguf.orig_shape." + name);
    if (m && m->isArray() && m->u64s.size() >= 2) {
        rows = (uint32_t)m->u64s[0];
        cols = 1;
        for (size_t i = 1; i < m->u64s.size(); i++) cols *= (uint32_t)m->u64s[i];
        return;
    }
    if (t->dims.size() <= 1) {
        rows = (uint32_t)(t->dims.empty() ? 1 : t->dims[0]);
        cols = 1;
        return;
    }
    if (t->dims.size() > 2) {
        // Untagged full-checkpoint GGUFs keep convs in their native [kx, ky, in, out] order.
        // PyTorch stores them [out, in, ky, kx], which is the same bytes read out-major.
        rows = (uint32_t)t->dims.back();
        cols = 1;
        for (size_t i = 0; i + 1 < t->dims.size(); i++) cols *= (uint32_t)t->dims[i];
        return;
    }
    rows = (uint32_t)t->dims[1];
    cols = (uint32_t)t->dims[0];
}

static const STTensor* findST(const std::string& name) {
    for (auto& e : gSTs) {
        if (!e.first.empty() && name.rfind(e.first, 0) == 0) {
            // prefixed lookup: only the file that owns this prefix
            return e.second->find(name.substr(e.first.size()));
        }
    }
    for (auto& e : gSTs) {
        const STTensor* t = e.second->find(name);
        if (t) return t;
    }
    return nullptr;
}
static void sdims(const std::string& name, uint32_t& rows, uint32_t& cols) {
    const STTensor* t = findST(name);
    if (!t) fail("missing safetensors tensor " + name);
    rows = (uint32_t)t->shape[0];
    cols = 1;
    for (size_t i = 1; i < t->shape.size(); i++) cols *= (uint32_t)t->shape[i];
}
static bool hasST(const std::string& name) { return findST(name) != nullptr; }

// dims of a weight that may live in the GGUF (Q8) or in a safetensors file (fp32)
// Route by which container holds the tensor: the GGUF also stores some UNet tensors as F16/F32.
static void linearDims(const std::string& name, uint32_t& rows, uint32_t& cols) {
    auto it = gDims.find(name);
    if (it != gDims.end()) {
        rows = it->second.first;
        cols = it->second.second;
        return;
    }
    if (gUNet.gg.find(name)) ggdims(name, rows, cols);
    else sdims(name, rows, cols);
}
static uint32_t wdimT(const std::string& name) {
    uint32_t r, c;
    linearDims(name, r, c);
    return r;
}

static void clearWeights() {
    gW.clear();
    gWQ.clear();
    gWS.clear();
    gDims.clear();
    gSTs.clear();
}

// ================================================================ weight loading


// ---------------------------------------------------------------- LoRA
// Kohya/PEFT LoRA pairs are merged into the resident UNet weights at load time:
// Q8_0 tensors dequant -> +scale*B@A -> requant, fp32 tensors are updated in place.
// DoRA checkpoints also carry dora_scale; it is currently ignored and the pair is used as
// a standard LoRA (applying the magnitude renormalisation on quantised weights broke outputs).
struct LoraPair {
    std::vector<float> down, up;
    float alpha = 0.f;
    uint32_t rank = 0, in = 0, out = 0;
};
static std::unordered_map<std::string, LoraPair> gLora;
static float gLoraScale = 1.0f;
static uint64_t gLoraApplied = 0;

static bool loraReplace(std::string& s, const char* a, const char* b) {
    size_t p = s.find(a);
    if (p == std::string::npos) return false;
    s.replace(p, std::strlen(a), b);
    return true;
}

// Kohya checkpoints come in two naming families. The diffusers one uses down_blocks_*;
// the LDM one encodes the walker key directly with underscores
// (input_blocks_4_1_transformer_blocks_0_attn1_to_q). Rebuild the dotted LDM key from the
// latter, keeping the tokens that themselves contain underscores.
static std::string ldmUnderscoreName(const std::string& s) {
    std::vector<std::string> tk, out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find('_', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) tk.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    for (size_t i = 0; i < tk.size(); i++) {
        const std::string& a = tk[i];
        const std::string b = i + 1 < tk.size() ? tk[i + 1] : std::string();
        const std::string c = i + 2 < tk.size() ? tk[i + 2] : std::string();
        if (a == "input" && b == "blocks") { out.push_back("input_blocks"); i++; }
        else if (a == "output" && b == "blocks") { out.push_back("output_blocks"); i++; }
        else if (a == "middle" && b == "block") { out.push_back("middle_block"); i++; }
        else if (a == "transformer" && b == "blocks") { out.push_back("transformer_blocks"); i++; }
        else if (a == "time" && b == "embed") { out.push_back("time_embed"); i++; }
        else if (a == "label" && b == "emb") { out.push_back("label_emb"); i++; }
        else if (a == "time" && b == "emb" && c == "proj") { out.push_back("time_emb_proj"); i += 2; }
        else if (a == "in" && b == "layers") { out.push_back("in_layers"); i++; }
        else if (a == "out" && b == "layers") { out.push_back("out_layers"); i++; }
        else if (a == "emb" && b == "layers") { out.push_back("emb_layers"); i++; }
        else if (a == "skip" && b == "connection") { out.push_back("skip_connection"); i++; }
        else if (a == "proj" && (b == "in" || b == "out")) { out.push_back("proj_" + b); i++; }
        else if (a == "to" && (b == "q" || b == "k" || b == "v" || b == "out")) {
            out.push_back("to_" + b);
            i++;
        } else {
            out.push_back(a);
        }
    }
    std::string r;
    for (auto& x : out) {
        if (!r.empty()) r += '.';
        r += x;
    }
    return r;
}

// lora_te1_* / lora_te2_* -> the te1./te2. keys used by loadOneTextEncoder.
static std::string loraTeToKey(const std::string& mod) {
    std::string pre, s;
    if (mod.rfind("lora_te1_", 0) == 0) { pre = "te1."; s = mod.substr(9); }
    else if (mod.rfind("lora_te2_", 0) == 0) { pre = "te2."; s = mod.substr(9); }
    else return {};
    std::vector<std::string> tk, out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find('_', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) tk.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    for (size_t i = 0; i < tk.size(); i++) {
        const std::string& a = tk[i];
        const std::string b = i + 1 < tk.size() ? tk[i + 1] : std::string();
        const std::string c = i + 2 < tk.size() ? tk[i + 2] : std::string();
        if (a == "text" && b == "model") { out.push_back("text_model"); i++; }
        else if (a == "text" && b == "projection") { out.push_back("text_projection"); i++; }
        else if (a == "self" && b == "attn") { out.push_back("self_attn"); i++; }
        else if ((a == "k" || a == "q" || a == "v" || a == "out") && b == "proj") {
            out.push_back(a + "_proj");
            i++;
        }
        else if (a == "layer" && (b == "norm1" || b == "norm2")) { out.push_back("layer_" + b); i++; }
        else if (a == "final" && b == "layer" && c == "norm") { out.push_back("final_layer_norm"); i += 2; }
        else if (a == "token" && b == "embedding") { out.push_back("token_embedding"); i++; }
        else if (a == "position" && b == "embedding") { out.push_back("position_embedding"); i++; }
        else if (a == "logit" && b == "scale") { out.push_back("logit_scale"); i++; }
        else if (a == "ln" && b == "final") { out.push_back("ln_final"); i++; }
        else out.push_back(a);
    }
    std::string r = pre;
    for (size_t i = 0; i < out.size(); i++) {
        if (i) r += '.';
        r += out[i];
    }
    return r;
}

// lora_unet_<diffusers module> -> LDM walker key (without .weight)
static std::string loraModuleToLdm(const std::string& mod) {
    std::string s = mod;
    std::string te = loraTeToKey(mod);
    if (!te.empty()) return te;
    const std::string pre = "lora_unet_";
    if (s.rfind(pre, 0) != 0) return {};
    s = s.substr(pre.size());
    if (s.rfind("input_blocks_", 0) == 0 || s.rfind("middle_block_", 0) == 0 ||
        s.rfind("output_blocks_", 0) == 0 || s.rfind("out_", 0) == 0 || s.rfind("time_embed_", 0) == 0 ||
        s.rfind("label_emb_", 0) == 0)
        return ldmUnderscoreName(s);
    auto digits = [&](size_t& p) {
        size_t q = p;
        while (q < s.size() && s[q] >= '0' && s[q] <= '9') q++;
        if (q == p) return -1;
        int v = std::atoi(s.substr(p, q - p).c_str());
        p = q;
        return v;
    };
    auto resSuffix = [](const std::string& t) -> std::string {
        if (t == "conv1") return "in_layers.2";
        if (t == "conv2") return "out_layers.3";
        if (t == "conv_shortcut") return "skip_connection";
        if (t == "time_emb_proj") return "emb_layers.1";
        return {};
    };
    std::string ldm;
    if (s.rfind("down_blocks_", 0) == 0) {
        size_t p = std::strlen("down_blocks_");
        int b = digits(p);
        if (b < 0) return {};
        if (s.compare(p, std::strlen("_attentions_"), "_attentions_") == 0) {
            p += std::strlen("_attentions_");
            int a = digits(p);
            if ((b != 1 && b != 2) || a < 0 || a > 1 || p >= s.size() || s[p] != '_') return {};
            p++;
            ldm = "input_blocks." + std::to_string(b == 1 ? 4 + a : 7 + a) + ".1";
            s = s.substr(p);
        } else if (s.compare(p, std::strlen("_resnets_"), "_resnets_") == 0) {
            p += std::strlen("_resnets_");
            int r = digits(p);
            if (b < 0 || b > 2 || r < 0 || r > 1 || p >= s.size() || s[p] != '_') return {};
            std::string suf = resSuffix(s.substr(p + 1));
            if (suf.empty()) return {};
            return "input_blocks." + std::to_string(1 + 3 * b + r) + ".0." + suf;
        } else if (s.compare(p, std::strlen("_downsamplers_0_conv"), "_downsamplers_0_conv") == 0) {
            if (b < 0 || b > 2) return {};
            return "input_blocks." + std::to_string(3 + 3 * b) + ".0.op";
        } else {
            return {};
        }
    } else if (s.rfind("mid_block_", 0) == 0) {
        size_t p = std::strlen("mid_block_");
        if (s.compare(p, std::strlen("attentions_"), "attentions_") == 0) {
            p += std::strlen("attentions_");
            int a = digits(p);
            if (a != 0 || p >= s.size() || s[p] != '_') return {};
            ldm = "middle_block.1";
            s = s.substr(p + 1);
        } else if (s.compare(p, std::strlen("resnets_"), "resnets_") == 0) {
            p += std::strlen("resnets_");
            int r = digits(p);
            if ((r != 0 && r != 1) || p >= s.size() || s[p] != '_') return {};
            std::string suf = resSuffix(s.substr(p + 1));
            if (suf.empty()) return {};
            return std::string("middle_block.") + (r == 0 ? "0" : "2") + "." + suf;
        } else {
            return {};
        }
    } else if (s.rfind("up_blocks_", 0) == 0) {
        size_t p = std::strlen("up_blocks_");
        int b = digits(p);
        if (b < 0) return {};
        if (s.compare(p, std::strlen("_attentions_"), "_attentions_") == 0) {
            p += std::strlen("_attentions_");
            int a = digits(p);
            if ((b != 0 && b != 1) || a < 0 || a > 2 || p >= s.size() || s[p] != '_') return {};
            p++;
            ldm = "output_blocks." + std::to_string(3 * b + a) + ".1";
            s = s.substr(p);
        } else if (s.compare(p, std::strlen("_resnets_"), "_resnets_") == 0) {
            p += std::strlen("_resnets_");
            int r = digits(p);
            if (b < 0 || b > 2 || r < 0 || r > 2 || p >= s.size() || s[p] != '_') return {};
            std::string suf = resSuffix(s.substr(p + 1));
            if (suf.empty()) return {};
            return "output_blocks." + std::to_string(3 * b + r) + ".0." + suf;
        } else if (s.compare(p, std::strlen("_upsamplers_0_conv"), "_upsamplers_0_conv") == 0) {
            if (b < 0 || b > 1) return {};
            return "output_blocks." + std::to_string(3 * b + 2) + ".2.conv";
        } else {
            return {};
        }
    } else if (s == "conv_in") {
        return "input_blocks.0.0";
    } else if (s == "conv_out") {
        return "out.2";
    } else if (s == "time_embedding_linear_1" || s == "time_embedding_linear_2") {
        return s.back() == '1' ? "time_embed.0" : "time_embed.2";
    } else if (s == "add_embedding_linear_1" || s == "add_embedding_linear_2") {
        return s.back() == '1' ? "label_emb.0.0" : "label_emb.0.2";
    } else {
        return {};
    }

    if (s == "proj_in" || s == "proj_out") return ldm + "." + s;
    if (s.rfind("transformer_blocks_", 0) != 0) return {};
    size_t p = std::strlen("transformer_blocks_");
    int k = digits(p);
    if (k < 0 || p >= s.size() || s[p] != '_') return {};
    std::string t = "transformer_blocks." + std::to_string(k) + s.substr(p);
    loraReplace(t, "_attn1_to_out_0", ".attn1.to_out.0");
    loraReplace(t, "_attn1_to_q", ".attn1.to_q");
    loraReplace(t, "_attn1_to_k", ".attn1.to_k");
    loraReplace(t, "_attn1_to_v", ".attn1.to_v");
    loraReplace(t, "_attn2_to_out_0", ".attn2.to_out.0");
    loraReplace(t, "_attn2_to_q", ".attn2.to_q");
    loraReplace(t, "_attn2_to_k", ".attn2.to_k");
    loraReplace(t, "_attn2_to_v", ".attn2.to_v");
    loraReplace(t, "_ff_net_0_proj", ".ff.net.0.proj");
    loraReplace(t, "_ff_net_2", ".ff.net.2");
    return ldm + "." + t;
}

static void loadLoraFile(const std::string& path, float userScale) {
    gLora.clear();
    gLoraScale = userScale;
    gLoraApplied = 0;
    Safetensors st;
    if (!st.ensureOpen(path)) fail("cannot open LoRA " + path + ": " + st.error());
    const std::string downSuf = ".lora_down.weight";
    uint32_t pairs = 0, skipped = 0;
    for (auto& t : st.tensors()) {
        if (t.name.size() <= downSuf.size() ||
            t.name.compare(t.name.size() - downSuf.size(), downSuf.size(), downSuf) != 0)
            continue;
        std::string mod = t.name.substr(0, t.name.size() - downSuf.size());
        std::string key = loraModuleToLdm(mod);
        if (key.empty()) { skipped++; continue; }
        const STTensor* ta = st.find(mod + ".lora_down.weight");
        const STTensor* tb = st.find(mod + ".lora_up.weight");
        if (!ta || !tb || ta->shape.empty() || tb->shape.empty()) { skipped++; continue; }
        LoraPair L;
        L.rank = (uint32_t)ta->shape[0];
        uint64_t dn = st.numel(*ta), un = st.numel(*tb);
        if (L.rank == 0 || L.rank > 256 || dn % L.rank || un % L.rank) { skipped++; continue; }
        L.in = (uint32_t)(dn / L.rank);
        L.out = (uint32_t)(un / L.rank);
        try {
            L.down = st.readF32(ta->name);
            L.up = st.readF32(tb->name);
        } catch (const std::exception& e) {
            fail(std::string("LoRA tensor read failed: ") + e.what());
        }
        if (L.down.size() != (size_t)L.rank * L.in || L.up.size() != (size_t)L.out * L.rank) { skipped++; continue; }
        if (const STTensor* al = st.find(mod + ".alpha")) {
            auto v = st.readF32(al->name);
            if (!v.empty()) L.alpha = v[0];
        }
        gLora[key + ".weight"] = std::move(L);
        pairs++;
    }
    std::printf("lora: %s -> %u modules mapped, %u skipped, scale %.2f\n", path.c_str(), pairs, skipped,
                userScale);
}

static void loraApplyF32(std::vector<float>& w, uint32_t rows, uint32_t cols, const LoraPair& L) {
    if (L.out != rows || L.in != cols || gLoraScale == 0.0f) return;
    const float scale = (L.alpha > 0.f ? L.alpha / (float)L.rank : 1.0f) * gLoraScale;
    for (uint32_t o = 0; o < rows; o++) {
        float* wr = &w[(size_t)o * cols];
        for (uint32_t r = 0; r < L.rank; r++) {
            float b = L.up[(size_t)o * L.rank + r] * scale;
            if (b == 0.0f) continue;
            const float* a = &L.down[(size_t)r * cols];
            for (uint32_t i = 0; i < cols; i++) wr[i] += b * a[i];
        }
    }
    gLoraApplied++;
}

static void loraApplyQ8(std::vector<int8_t>& q, std::vector<float>& sc, uint32_t rows, uint32_t cols,
                        const LoraPair& L) {
    if (L.out != rows || L.in != cols || gLoraScale == 0.0f) return;
    const float scale = (L.alpha > 0.f ? L.alpha / (float)L.rank : 1.0f) * gLoraScale;
    std::vector<float> row(cols);
    const uint32_t blocks = cols / 32;
    for (uint32_t o = 0; o < rows; o++) {
        int8_t* qr = &q[(size_t)o * cols];
        float* sr = &sc[(size_t)o * blocks];
        for (uint32_t i = 0; i < cols; i++) row[i] = (float)qr[i] * sr[i >> 5];
        for (uint32_t r = 0; r < L.rank; r++) {
            float b = L.up[(size_t)o * L.rank + r] * scale;
            if (b == 0.0f) continue;
            const float* a = &L.down[(size_t)r * cols];
            for (uint32_t i = 0; i < cols; i++) row[i] += b * a[i];
        }
        for (uint32_t b = 0; b < blocks; b++) {
            float mx = 0.0f;
            for (uint32_t i = 0; i < 32; i++) mx = std::max(mx, std::fabs(row[b * 32 + i]));
            float s = mx > 0.0f ? mx / 127.0f : 0.0f;
            float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
            sr[b] = s;
            for (uint32_t i = 0; i < 32; i++) {
                float x = row[b * 32 + i] * inv;
                int v = (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
                qr[b * 32 + i] = (int8_t)std::max(-127, std::min(127, v));
            }
        }
    }
    gLoraApplied++;
}

static void loadGgufUnet() {
    GgufFile& gg = gUNet.gg;
    gLoraApplied = 0;
    uint64_t pending = 0, q8n = 0, f32n = 0;
    for (auto& t : gg.tensors()) {
        // tensors are written straight into the mapped staging ring: a separate host vector per
        // tensor would double the commit footprint of a 2.5 GB upload on a 16 GB box
        auto lora = gLora.find(t.name);
        // 1-D Q8 tensors are biases/norms. The conv/linear paths read those from the fp32 map,
        // and dequantizing a few hundred values is cheaper than teaching every kernel a Q8 bias.
        bool q8Vector = t.type == GgmlType::Q8_0 && t.dims.size() <= 1;
        if (t.type == GgmlType::Q8_0 && !q8Vector) {
            uint64_t ne = t.nElems(), nb = ne / 32;
            BufRef bq = G.paramArena.alloc((ne + 255) & ~255ull);
            BufRef bs = G.paramArena.alloc((nb * 4 + 255) & ~255ull);
            if (lora != gLora.end()) {
                uint32_t rows = 0, cols = 0;
                ggdims(t.name, rows, cols);
                std::vector<int8_t> q(ne);
                std::vector<float> sc(nb);
                std::string err;
                if (!gg.readQ8(t.name, q, sc, &err)) fail(err + " (" + t.name + ")");
                loraApplyQ8(q, sc, rows, cols, lora->second);
                // stageUpload copies immediately; the two beginUpload slots below can be
                // invalidated by a staging flush between the calls, which silently zeroed
                // parts of the merged payload even at --lora-scale 0.
                G.eng.stageUpload(bq, q.data(), (uint64_t)ne);
                G.eng.stageUpload(bs, sc.data(), (uint64_t)nb * 4);
            } else {
                int8_t* qd = (int8_t*)G.eng.beginUpload(bq, ne);
                float* sd = (float*)G.eng.beginUpload(bs, nb * 4);
                std::string err;
                if (!gg.readQ8Into(t.name, qd, sd, &err)) fail(err + " (" + t.name + ")");
            }
            gWQ[t.name] = bq;
            gWS[t.name] = bs;
            q8n++;
            pending += ne + nb * 4;
        } else {
            std::string err;
            auto v = gg.readFloat(t.name, &err);
            if (v.empty()) fail(err + " (" + t.name + ")");
            if (lora != gLora.end()) {
                uint32_t rows = 0, cols = 0;
                ggdims(t.name, rows, cols);
                if (v.size() == (size_t)rows * cols) loraApplyF32(v, rows, cols, lora->second);
            }
            BufRef b = G.paramArena.alloc((v.size() * 4 + 255) & ~255ull);
            G.eng.stageUpload(b, v.data(), v.size() * 4);
            gW[t.name] = b;
            f32n++;
            pending += v.size() * 4;
        }
        if (pending >= (96ull << 20)) {
            G.eng.flushUploads();
            pending = 0;
        }
    }
    G.eng.flushUploads();
    std::printf("unet weights: %llu Q8_0 + %llu fp (param arena %.0f MB)\n", (unsigned long long)q8n,
                (unsigned long long)f32n, G.paramArena.used() / 1048576.0);
    if (!gLora.empty()) std::printf("lora: applied to %llu tensors\n", (unsigned long long)gLoraApplied);
}

static bool containsPrefix(const Safetensors& st, const char* p) {
    for (auto& t : st.tensors())
        if (t.name.rfind(p, 0) == 0) return true;
    return false;
}

// Tensors are copied with one prefix swapped for another, so a diffusers component, a
// single-file checkpoint and a bare UNet file all land under the names the walkers use.
static void uploadF32(const std::string& key, std::vector<float>& v, uint64_t& pending) {
    BufRef b = G.paramArena.alloc((v.size() * 4 + 255) & ~255ull);
    G.eng.stageUpload(b, v.data(), v.size() * 4);
    gW[key] = b;
    pending += v.size() * 4;
    if (pending >= (96ull << 20)) {
        G.eng.flushUploads();
        pending = 0;
    }
}

// which slice of a file to take
enum class Part { All, Unet, ClipL, ClipG, ClipGSingleFile, Vae, Projection };

static bool wantTensor(const std::string& n, Part part) {
    switch (part) {
        case Part::All:
            return n.rfind("first_stage_model.encoder", 0) != 0 && n.rfind("vae.encoder", 0) != 0;
        case Part::Unet:
            return n.rfind("model.diffusion_model.", 0) == 0;
        case Part::ClipL:
            return n.rfind("text_encoder.", 0) == 0;
        case Part::ClipG:
            return n.rfind("text_encoder_2.", 0) == 0;
        case Part::ClipGSingleFile:
            return n.rfind("conditioner.embedders.1.model.", 0) == 0;
        case Part::Projection:
            return n.rfind("text_projection.", 0) == 0;
        case Part::Vae:
            // a diffusers export keeps the bare decoder./encoder./post_quant_conv. names with no
            // vae. prefix. The encoder half used to be dropped here, which capped img2img at a
            // pixel-seeded sampler; keep it so VAEEncode has weights to run with.
            if (n.rfind("first_stage_model.", 0) == 0 || n.rfind("vae.", 0) == 0) return true;
            return n.rfind("decoder.", 0) == 0 || n.rfind("encoder.", 0) == 0 ||
                   n.rfind("post_quant_conv.", 0) == 0 || n.rfind("quant_conv.", 0) == 0;
    }
    return false;
}

static void loadSafetensors(Safetensors& st, const std::string& from, const std::string& to, Part part,
                            const std::string& label) {
    uint64_t pending = 0, n = 0;
    // `from` is a real slice prefix only when the file actually carries it (checkpoints); for
    // diffusers components the caller passes a virtual alias ("vae." on decoder.* keys), and the
    // Part filter does the selection there.
    bool havePrefix = false;
    if (!from.empty()) {
        for (auto& t : st.tensors())
            if (t.name.rfind(from, 0) == 0) {
                havePrefix = true;
                break;
            }
    }
    for (auto& t : st.tensors()) {
        if (havePrefix && t.name.rfind(from, 0) != 0) continue;  // slice a checkpoint, don't load all of it
        if (!wantTensor(t.name, part)) continue;
        if (t.dtype != "F32" && t.dtype != "F16" && t.dtype != "BF16") continue;  // position_ids and friends
        std::vector<float> v;
        try {
            v = st.readF32(t.name);
        } catch (const std::exception& e) {
            fail(std::string("read ") + t.name + ": " + e.what());
        }
        std::string key = t.name;
        if (!from.empty() && key.rfind(from, 0) == 0) key = to + key.substr(from.size());
        uint32_t rows = t.shape.empty() ? 1u : (uint32_t)t.shape[0];
        uint64_t cols = 1;
        for (size_t i = 1; i < t.shape.size(); i++) cols *= t.shape[i];
        gDims[key] = {rows, (uint32_t)cols};
        // Text-encoder LoRA deltas: lora_te1_* -> te1.text_model.*. Some repos store the
        // encoders without the text_model. segment, so accept that alias too.
        // key already ends in ".weight", exactly like the gLora keys.
        auto loraIt = gLora.find(key);
        if (loraIt == gLora.end() && key.rfind("te1.text_model.", 0) == 0)
            loraIt = gLora.find("te1." + key.substr(std::strlen("te1.text_model.")));
        if (loraIt == gLora.end() && key.rfind("te2.text_model.", 0) == 0)
            loraIt = gLora.find("te2." + key.substr(std::strlen("te2.text_model.")));
        if (loraIt != gLora.end() && t.shape.size() == 2) loraApplyF32(v, rows, (uint32_t)cols, loraIt->second);
        uploadF32(key, v, pending);
        n++;
    }
    G.eng.flushUploads();
    std::printf("weights loaded (%s): %zu tensors, param arena %.0f MB\n", label.c_str(), n,
                G.paramArena.used() / 1048576.0);
    if (std::getenv("OFC_DBG_KEYS")) {
        int shown = 0;
        for (auto& kv : gW)
            if (kv.first.find("fc1") != std::string::npos && shown++ < 3) std::printf("  key: %s\n", kv.first.c_str());
        std::printf("  layer0 keys: %d, exact=%d\n", (int)std::count_if(gW.begin(), gW.end(), [](const auto& kv) {
                        return kv.first.find("layers.0.mlp") != std::string::npos;
                    }),
                    (int)gW.count("te1.text_model.encoder.layers.0.mlp.fc1.weight"));
    }
}

// Loads the UNet from a GGUF (kept quantized, dequantized in-shader) or a safetensors UNet.
static void loadUnet(const GenParams& p) {
    const std::string& f = p.models.unet;
    bool gguf = f.size() > 5 && f.compare(f.size() - 5, 5, ".gguf") == 0;
    if (gguf) {
        std::string err;
        if (!gUNet.gg.open(f, &err)) fail("cannot open UNet " + f + ": " + err);
        bool full = gUNet.gg.find("model.diffusion_model.input_blocks.0.0.weight") != nullptr;
        if (full && !gUNet.gg.open(f, &err, "model.diffusion_model."))
            fail("cannot reopen UNet " + f + ": " + err);
        gUNet.kind = 1;
        gUNet.path = f;
        std::printf("unet: %s (%zu tensors, gguf%s)\n", f.c_str(), gUNet.gg.tensorCount(),
                    full ? ", full checkpoint" : "");
        loadGgufUnet();
        gArch = hasAnyW("label_emb.0.0.weight") ? 2 : 1;
        std::printf("arch: %s\n", gArch == 1 ? "SD1.5" : "SDXL");
        return;
    }
    if (!gUNet.st.ensureOpen(f)) fail("cannot open UNet " + f + ": " + gUNet.st.error());
    gUNet.kind = 2;
    gUNet.path = f;
    bool checkpoint = containsPrefix(gUNet.st, "model.diffusion_model.");
    std::printf("unet: %s (%zu tensors, safetensors%s)\n", f.c_str(), gUNet.st.tensors().size(),
                checkpoint ? ", full checkpoint" : "");
    loadSafetensors(gUNet.st, checkpoint ? "model.diffusion_model." : "", "", checkpoint ? Part::Unet : Part::All,
                    "unet");
    gArch = hasAnyW("label_emb.0.0.weight") ? 2 : 1;
    std::printf("arch: %s\n", gArch == 1 ? "SD1.5" : "SDXL");
}

// Text encoders, one at a time. `phase` 0 = CLIP-L, 1 = CLIP-G; whichever pass runs rebinds the
// weight arena first, because both encoders together do not fit in the memory an 8 GB card leaves
// free. Every pass also registers its file in gSTs, which is only used for shape lookups.
static void loadOneTextEncoder(const GenParams& p, int phase) {
    const std::string& path = phase == 0 ? p.models.te1 : p.models.te2;
    const char* what = phase == 0 ? "CLIP-L" : "CLIP-G";
    Safetensors& st = phase == 0 ? gTe1.st : gTe2.st;
    if (!st.ensureOpen(path)) fail(std::string("cannot open ") + what + " " + path + ": " + st.error());
    std::error_code ec;
    uint64_t fileBytes = std::filesystem::file_size(path, ec);
    if (ec) fileBytes = 0;
    G.paramBytes = std::min<uint64_t>(std::max<uint64_t>(fileBytes * 2 / 1048576 + 512, 512), phase == 0 ? 2048 : 6144) << 20;
    clearWeights();
    G.eng.initArenas(192ull << 20, 192ull << 20);  // CLIP attention peaks near 77x77x1280
    G.paramArena.init(G.eng.device(), G.eng.physDevice(), G.paramBytes, true, false);
    std::printf("%s arena: %.0f MB\n", what, G.paramBytes / 1048576.0);

    if (containsPrefix(st, "cond_stage_model.transformer.")) {
        // SD1.5 single-file checkpoint: a bare CLIP-L under the LDM prefix (one extra segment).
        // Must be tested before the text_model.* branch: containsPrefix is a substring match and
        // "text_model." is inside "cond_stage_model.transformer.text_model.".
        if (phase != 0) fail("SD1.5 checkpoints carry a single text encoder; CLIP-G is not available");
        loadSafetensors(st, "cond_stage_model.transformer.text_model.", "te1.text_model.", Part::All, what);
    } else if (containsPrefix(st, "text_encoder.")) {
        loadSafetensors(st, phase == 0 ? "text_encoder." : "text_encoder_2.", phase == 0 ? "te1." : "te2.",
                        phase == 0 ? Part::ClipL : Part::ClipG, what);
    } else if (containsPrefix(st, "text_model.")) {
        loadSafetensors(st, "text_model.", phase == 0 ? "te1.text_model." : "te2.text_model.", Part::All, what);
        if (phase == 1 && containsPrefix(st, "text_projection."))
            loadSafetensors(st, "text_projection.", "te2.text_projection.", Part::Projection, "clip-g projection");
    } else if (containsPrefix(st, "conditioner.embedders.0.")) {
        loadSafetensors(st, "conditioner.embedders.0.transformer.", "te1.", Part::ClipL, "clip-l (checkpoint)");
        if (phase == 1) loadSafetensors(st, "conditioner.embedders.1.model.", "te2.text_model.", Part::ClipGSingleFile,
                                        "clip-g (checkpoint)");
    } else {
        fail(std::string("don't know how to read ") + what + " from " + path +
             " (expected text_encoder.*, text_model.* or conditioner.embedders.* keys)");
    }
    gSTs.push_back({phase == 0 ? "te1." : "te2.", &st});
}

// VAE for the decode pass (diffusers component or the first_stage_model slice of a checkpoint).
static void replaceAllStr(std::string& s, const std::string& a, const std::string& b) {
    size_t p = 0;
    while ((p = s.find(a, p)) != std::string::npos) {
        s.replace(p, a.size(), b);
        p += b.size();
    }
}

// LDM VAE checkpoints name the same modules differently from diffusers: mid.block_1 ->
// mid_block.resnets.0, attention q -> to_q, nin_shortcut -> conv_shortcut, and decoder.up is
// stored in the opposite order. Rebuild the diffusers key the VAE walker expects.
static std::string ldmVaeKey(const std::string& name, int upBlocks) {
    std::string s = name;
    auto mapBlocks = [&](const char* tag, bool reverse, const char* dst) {
        std::string pre = std::string(tag) + ".";
        if (s.rfind(pre, 0) != 0) return;
        size_t p = pre.size();
        size_t q = s.find('.', p);
        if (q == std::string::npos) return;
        int i = std::atoi(s.substr(p, q - p).c_str());
        std::string rest = s.substr(q + 1);
        int oi = reverse ? upBlocks - 1 - i : i;
        std::string np = std::string(dst) + "_blocks." + std::to_string(oi) + ".";
        if (rest.rfind("block.", 0) == 0) {
            size_t r = rest.find('.', 6);
            if (r == std::string::npos) return;
            int j = std::atoi(rest.substr(6, r - 6).c_str());
            s = np + "resnets." + std::to_string(j) + "." + rest.substr(r + 1);
        } else if (rest.rfind("downsample.conv.", 0) == 0) {
            s = np + "downsamplers.0.conv." + rest.substr(16);
        } else if (rest.rfind("upsample.conv.", 0) == 0) {
            s = np + "upsamplers.0.conv." + rest.substr(14);
        }
    };
    mapBlocks("encoder.down", false, "encoder.down");
    mapBlocks("decoder.up", true, "decoder.up");
    replaceAllStr(s, "mid.block_1.", "mid_block.resnets.0.");
    replaceAllStr(s, "mid.block_2.", "mid_block.resnets.1.");
    replaceAllStr(s, "mid.attn_1.", "mid_block.attentions.0.");
    replaceAllStr(s, "nin_shortcut", "conv_shortcut");
    replaceAllStr(s, ".norm_out.", ".conv_norm_out.");
    replaceAllStr(s, ".norm.", ".group_norm.");
    replaceAllStr(s, ".q.", ".to_q.");
    replaceAllStr(s, ".k.", ".to_k.");
    replaceAllStr(s, ".v.", ".to_v.");
    replaceAllStr(s, ".proj_out.", ".to_out.0.");
    return s;
}

static void loadVaeWeights(const GenParams& p) {
    bool gguf = p.models.vae.size() > 5 && p.models.vae.compare(p.models.vae.size() - 5, 5, ".gguf") == 0;
    if (gguf) {
        // Full-checkpoint GGUF: the VAE sits next to the UNet as first_stage_model.*.
        GgufFile gg;
        std::string err;
        if (!gg.open(p.models.vae, &err)) fail("cannot open VAE " + p.models.vae + ": " + err);
        bool full = gg.find("first_stage_model.decoder.conv_in.weight") != nullptr;
        if (full && !gg.open(p.models.vae, &err, "first_stage_model."))
            fail("cannot reopen VAE " + p.models.vae + ": " + err);
        if (!full && !gg.find("decoder.conv_in.weight"))
            fail("VAE GGUF has no decoder/first_stage_model tensors: " + p.models.vae);
        int upBlocks = 4;
        if (full) {
            for (auto& t : gg.tensors()) {
                size_t p = t.name.find("decoder.up.");
                if (p != 0) continue;
                size_t q = t.name.find('.', p + 11);
                if (q != std::string::npos) upBlocks = std::max(upBlocks, std::atoi(t.name.substr(p + 11, q - p - 11).c_str()) + 1);
            }
        }
        uint64_t pending = 0;
        size_t n = 0;
        for (auto& t : gg.tensors()) {
            auto v = gg.readFloat(t.name, &err);
            if (v.empty()) fail("read VAE " + t.name + ": " + err);
            std::string key = full ? ldmVaeKey(t.name, upBlocks) : t.name;
            uint32_t rows;
            uint64_t cols = 1;
            if (t.dims.size() > 2) {
                rows = (uint32_t)t.dims.back();
                for (size_t i = 0; i + 1 < t.dims.size(); i++) cols *= t.dims[i];
            } else if (t.dims.size() == 2) {
                rows = (uint32_t)t.dims[1];
                cols = t.dims[0];
            } else {
                rows = (uint32_t)(t.dims.empty() ? 1 : t.dims[0]);
            }
            gDims[key] = {rows, (uint32_t)cols};
            uploadF32(key, v, pending);
            n++;
        }
        G.eng.flushUploads();
        std::printf("weights loaded (vae gguf): %zu tensors, param arena %.0f MB\n", n,
                    G.paramArena.used() / 1048576.0);
        return;
    }
    if (!gVae.st.ensureOpen(p.models.vae)) fail("cannot open VAE " + p.models.vae + ": " + gVae.st.error());
    if (containsPrefix(gVae.st, "first_stage_model.")) {
        loadSafetensors(gVae.st, "first_stage_model.", "", Part::Vae, "vae (checkpoint)");
    } else {
        loadSafetensors(gVae.st, "vae.", "", Part::Vae, "vae");
    }
    gSTs.push_back({"", &gVae.st});
}

// ================================================================ dispatch helpers

struct Pg {
    uint32_t M, N, K, mode, sA, sB, sC, batch, lda, ldb, ldc, pad;
};
struct Pnd {
    uint32_t n, d0, d1, d2, d3, os0, os1, os2, os3, ss0, ss1, ss2, ss3, mode;
    float scale;
};
struct P2 {
    uint32_t a, b;
};

static void gemmNt(const BufRef& A, const BufRef& B, const BufRef& C, uint32_t M, uint32_t N, uint32_t K) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    gDisp("gemm_nt", {A, B, C}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
}

static void gemmNtQ8(const BufRef& A, const BufRef& Bq, const BufRef& C, const BufRef& Bs, uint32_t M, uint32_t N,
                     uint32_t K) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    // BM=128 halves weight/dequant traffic per output and wins once M is large.
    if (M >= 1024) {
        gDisp("gemm_nt_q8_v_bm128", {A, Bq, C, Bs}, &pc, ceilDiv(N, 64), ceilDiv(M, 128), 1);
    } else {
        gDisp("gemm_nt_q8_v", {A, Bq, C, Bs}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
    }
}

static void gemmNtQ8Bias(const BufRef& A, const BufRef& Bq, const BufRef& C, const BufRef& Bs,
                         const BufRef& bias, uint32_t M, uint32_t N, uint32_t K) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    if (M >= 1024) {
        gDisp("gemm_nt_q8_v_bm128_b", {A, Bq, C, Bs, bias}, &pc, ceilDiv(N, 64), ceilDiv(M, 128), 1);
    } else {
        gDisp("gemm_nt_q8_v_b", {A, Bq, C, Bs, bias}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
    }
}

static void gemmNn(const BufRef& A, const BufRef& B, const BufRef& C, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t ldc) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, ldc, 0};
    gDisp("gemm_nn", {A, B, C}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
}

static void gemmNnQ8(const BufRef& Aq, const BufRef& As, const BufRef& B, const BufRef& C, uint32_t M, uint32_t N,
                     uint32_t K, uint32_t ldc) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, ldc, 0};
    gDisp("gemm_nn_q8_v", {Aq, B, C, As}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
}

static BufRef copyNd(const BufRef& src, uint32_t n, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t os0, uint32_t os1,
                     uint32_t os2, uint32_t ss0, uint32_t ss1, uint32_t ss2, float scale = 1.0f) {
    BufRef dst = allocN(n);
    Pnd pc{n, d0, d1, d2, 1, os0, os1, os2, 0, ss0, ss1, ss2, 0, 0, scale};
    gDisp("copy_nd", {dst, src}, &pc, ceilDiv(n, 256), 1, 1);
    return dst;
}

static void copyInto(const BufRef& dst, uint64_t dstOff, const BufRef& src, uint64_t bytes) {
    BufRef d{dst.buf, dst.offset + dstOff, bytes};
    BufRef s{src.buf, src.offset, bytes};
    if (gArenaPlan) return;
    G.eng.copyBuf(d, s);
}

// device-side trace: snapshot a buffer into the scratch arena (copy recorded in order),
// downloaded after waitStep by flushTrace()
static bool gTrace = false;
static bool gVaeTraceSub = false;  // OFC_VAE_TRACE_SUB=1: dump the first resnet's sub-steps
static const char* gDbSel = "";  // dump the internals of this one stage only
struct TrEnt {
    std::string name;
    BufRef b;
    uint64_t floats;
};
static std::vector<TrEnt> gTr;
static void tr(const std::string& name, const BufRef& b, uint64_t floats) {
    if (!gTrace) return;
    BufRef c = G.scratchArena.alloc((floats * 4 + 255) & ~255ull);
    copyInto(c, 0, b, floats * 4);
    gTr.push_back({name, c, floats});
}
static void flushTrace() {
    for (auto& e : gTr) {
        std::vector<float> v(e.floats);
        G.eng.downloadNow(e.b, v.data(), v.size() * 4);
        FILE* f = std::fopen(("dbg_" + e.name + ".f32").c_str(), "wb");
        std::fwrite(v.data(), 4, v.size(), f);
        std::fclose(f);
    }
    gTr.clear();
}

static BufRef add2Into(const BufRef& a, const BufRef& b, uint64_t n, const BufRef& c) {
    P2 pc{(uint32_t)n, 0};
    gDisp("add2", {c, a, b}, &pc, ceilDiv(n, 256), 1, 1);
    return c;
}

static BufRef add2(const BufRef& a, const BufRef& b, uint64_t n) {
    return add2Into(a, b, n, allocN(n));
}

static BufRef mul2(const BufRef& a, const BufRef& b, uint64_t n) {
    BufRef c = allocN(n);
    struct {
        uint32_t n, d0, d1, d2, d3, as0, as1, as2, as3, bs0, bs1, bs2, bs3, mode;
    } pc;
    std::memset(&pc, 0, sizeof pc);
    pc.n = (uint32_t)n;
    pc.d0 = (uint32_t)n;
    pc.d1 = 1;
    pc.d2 = 1;
    pc.d3 = 1;
    pc.as0 = 1;
    pc.bs0 = 1;
    pc.mode = 1;
    gDisp("ew2", {c, a, b}, &pc, ceilDiv(n, 256), 1, 1);
    return c;
}

static BufRef unaryInto(const BufRef& x, uint64_t n, uint32_t mode, const BufRef& y) {
    struct {
        uint32_t n, mode;
    } pc{(uint32_t)n, mode};
    gDisp("unary_fwd", {x, y}, &pc, ceilDiv(n, 256), 1, 1);
    return y;
}

static BufRef unary(const BufRef& x, uint64_t n, uint32_t mode) {
    return unaryInto(x, n, mode, allocN(n));
}
static BufRef silu(const BufRef& x, uint64_t n) { return unary(x, n, 2); }
static BufRef siluInto(const BufRef& x, uint64_t n, const BufRef& y) { return unaryInto(x, n, 2, y); }
static BufRef geluErf(const BufRef& x, uint64_t n) { return unary(x, n, 4); }

static BufRef softmax(const BufRef& x, uint32_t rows, uint32_t cols) {
    BufRef y = allocN((uint64_t)rows * cols);
    P2 pc{rows, cols};
    gDisp("softmax_fwd", {x, y}, &pc, rows, 1, 1);
    return y;
}

static void biasAddLast(const BufRef& y, const BufRef& b, uint32_t rows, uint32_t d) {
    struct {
        uint32_t n, d0, d1, d2, d3, as0, as1, as2, as3, bs0, bs1, bs2, bs3, mode;
    } pc;
    std::memset(&pc, 0, sizeof pc);
    pc.n = rows * d;
    pc.d0 = d;
    pc.d1 = rows;
    pc.d2 = 1;
    pc.d3 = 1;
    pc.as0 = 1;
    pc.as1 = d;
    pc.bs0 = 1;
    pc.mode = 0;
    gDisp("ew2", {y, y, b}, &pc, ceilDiv((uint64_t)rows * d, 256), 1, 1);
}

static void biasAddCh(const BufRef& y, const BufRef& b, uint32_t C, uint32_t P) {
    struct {
        uint32_t C, P;
    } pb{C, P};
    gDisp("bias_add", {y, b}, &pb, ceilDiv((uint64_t)C * P, 256), 1, 1);
}

static void scaleInPlace(const BufRef& x, uint64_t n, float s) {
    Pnd pc{(uint32_t)n, (uint32_t)n, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, s};
    gDisp("copy_nd", {x, x}, &pc, ceilDiv(n, 256), 1, 1);
}

static BufRef transpose2(const BufRef& x, uint32_t outer, uint32_t inner) {
    return copyNd(x, outer * inner, inner, outer, 1, outer, 1, 0, 1, inner, 0);
}

// linear over a weight set that may be Q8 (gguf) or fp32 (safetensors)
static BufRef linear(const BufRef& x, const std::string& p, uint32_t rows) {
    uint32_t nout, nin;
    linearDims(p + ".weight", nout, nin);
    BufRef y = allocN((uint64_t)rows * nout);
    if (isQ8(p + ".weight")) {
        if (hasW(p + ".bias")) {
            gemmNtQ8Bias(x, WQ(p + ".weight"), y, WS(p + ".weight"), W(p + ".bias"), rows, nout, nin);
            return y;
        }
        gemmNtQ8(x, WQ(p + ".weight"), y, WS(p + ".weight"), rows, nout, nin);
    } else {
        gemmNt(x, W(p + ".weight"), y, rows, nout, nin);
    }
    if (hasW(p + ".bias")) biasAddLast(y, W(p + ".bias"), rows, nout);
    return y;
}

static BufRef layernorm(const BufRef& x, const std::string& p, uint32_t rows, uint32_t d) {
    BufRef y = allocN((uint64_t)rows * d);
    struct {
        uint32_t rows, d;
        float eps;
        uint32_t affine;
    } pc{rows, d, 1e-5f, 1};
    gDisp("layernorm_fwd", {x, W(p + ".weight"), W(p + ".bias"), y}, &pc, rows, 1, 1);
    return y;
}

static BufRef gnInto(const BufRef& x, const std::string& p, uint32_t C, uint32_t P, uint32_t groups, float eps,
                     const BufRef& y, uint32_t act = 0) {
    struct {
        uint32_t C, P, G;
        float eps;
        uint32_t act;
    } pc{C, P, groups, eps, act};
    gDisp("groupnorm", {x, W(p + ".weight"), W(p + ".bias"), y}, &pc, groups, 1, 1);
    return y;
}

static BufRef gn(const BufRef& x, const std::string& p, uint32_t C, uint32_t P, uint32_t groups = 32,
                 float eps = 1e-5f) {
    return gnInto(x, p, C, P, groups, eps, allocN((uint64_t)C * P));
}

// GroupNorm with SiLU in the same pass: the two used to be a norm dispatch plus a full
// read-modify-write of the tensor, which is pure bandwidth on a 180 GB/s card.
static BufRef gnSilu(const BufRef& x, const std::string& p, uint32_t C, uint32_t P, uint32_t groups = 32,
                     float eps = 1e-5f) {
    return gnInto(x, p, C, P, groups, eps, allocN((uint64_t)C * P), 1);
}

// ================================================================ convolutions

static BufRef gCol;
static const uint64_t kColFloats = 12000000;  // 48 MB reusable im2col buffer

static BufRef conv3RawInto(const BufRef& in, const std::string& wt, uint32_t cin, uint32_t cout, uint32_t H,
                           uint32_t Wd, const BufRef& out) {
    uint32_t P = H * Wd, K = cin * 9;
    uint32_t rows = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(H, kColFloats / (uint64_t)K / Wd));
    for (uint32_t r0 = 0; r0 < H; r0 += rows) {
        uint32_t rr = std::min(rows, H - r0), np = rr * Wd;
        struct {
            uint32_t C, H, W, row0, rows;
        } pc{cin, H, Wd, r0, rr};
        BufRef col{gCol.buf, gCol.offset, (uint64_t)K * np * 4};
        gDisp("im2col", {in, col}, &pc, ceilDiv((uint64_t)K * np, 256), 1, 1);
        BufRef cs{out.buf, out.offset + (uint64_t)r0 * Wd * 4, out.size - (uint64_t)r0 * Wd * 4};
        if (isQ8(wt)) gemmNnQ8(WQ(wt), WS(wt), col, cs, cout, np, K, P);
        else gemmNn(W(wt), col, cs, cout, np, K, P);
    }
    std::string biasName = wt;  // callers pass "<prefix>.weight"; the bias shares that prefix
    biasName.erase(biasName.size() - 7);
    biasName += ".bias";
    biasAddCh(out, W(biasName), cout, P);
    return out;
}

// out[cout,P] = wt[cout, cin*9] * col[cin*9, P]; row-chunked to bound the col buffer
static BufRef conv3Raw(const BufRef& in, const std::string& wt, uint32_t cin, uint32_t cout, uint32_t H, uint32_t Wd) {
    return conv3RawInto(in, wt, cin, cout, H, Wd, allocN((uint64_t)cout * H * Wd));
}

// off = -1 keeps the symmetric pad-1 ring the LDM downsamplers expect; off = 0 pads only the
// right/bottom edge, which is what diffusers' AutoencoderKL Downsample2D does for the VAE encoder.
static BufRef conv3s2(const BufRef& in, const std::string& p, uint32_t H, uint32_t Wd, int off = -1) {
    uint32_t cout, kCols;  // conv weight dims are (cout, cin*k)
    linearDims(p + ".weight", cout, kCols);
    uint32_t cin = kCols / 9;
    uint32_t OH = H / 2, OW = Wd / 2, K = cin * 9, P = OH * OW;
    BufRef out = allocN((uint64_t)cout * P);
    struct {
        uint32_t C, H, W, OH, OW;
        int off;
    } pc{cin, H, Wd, OH, OW, off};
    BufRef col{gCol.buf, gCol.offset, (uint64_t)K * P * 4};
    gDisp("im2col_s2", {in, col}, &pc, ceilDiv((uint64_t)K * P, 256), 1, 1);
    if (isQ8(p + ".weight")) gemmNnQ8(WQ(p + ".weight"), WS(p + ".weight"), col, out, cout, P, K, P);
    else gemmNn(W(p + ".weight"), col, out, cout, P, K, P);
    biasAddCh(out, W(p + ".bias"), cout, P);
    return out;
}

static BufRef conv1(const BufRef& in, const std::string& p, uint32_t cin, uint32_t cout, uint32_t P) {
    BufRef out = allocN((uint64_t)cout * P);
    if (isQ8(p + ".weight")) {
        gemmNnQ8(WQ(p + ".weight"), WS(p + ".weight"), in, out, cout, P, cin, P);
    } else {
        gemmNn(W(p + ".weight"), in, out, cout, P, cin, P);
    }
    biasAddCh(out, W(p + ".bias"), cout, P);
    return out;
}

static BufRef up2(const BufRef& x, uint32_t C, uint32_t H, uint32_t Wd) {
    BufRef y = allocN((uint64_t)C * (H * 2) * (Wd * 2));
    struct {
        uint32_t C, H, W;
    } pc{C, H, Wd};
    gDisp("upsample2x", {x, y}, &pc, ceilDiv((uint64_t)C * H * 2 * Wd * 2, 256), 1, 1);
    return y;
}

// ================================================================ attention

static BufRef gAttnS;
static uint64_t gSCap = 0;

static void ensureAttnScratch(uint64_t floats) {
    if (gSCap >= floats) return;
    gAttnS = G.scratchArena.alloc(floats * 4);
    gSCap = floats;
}

static BufRef headsView(const BufRef& x, uint32_t T, uint32_t C, uint32_t heads, uint32_t hOff, uint32_t hCnt,
                        float scale) {
    uint32_t hd = C / heads;
    BufRef src{x.buf, x.offset + (uint64_t)hOff * hd * 4, x.size - (uint64_t)hOff * hd * 4};
    return copyNd(src, hCnt * T * hd, hd, T, hCnt, 1, hd, T * hd, 1, C, hd, scale);
}

static void headsMergeInto(const BufRef& dst, const BufRef& x, uint32_t T, uint32_t C, uint32_t heads, uint32_t hOff,
                           uint32_t hCnt) {
    uint32_t hd = C / heads;
    BufRef d{dst.buf, dst.offset + (uint64_t)hOff * hd * 4, dst.size - (uint64_t)hOff * hd * 4};
    Pnd pc{hCnt * T * hd, hd, T, hCnt, 1, 1, C, hd, 0, 1, hd, T * hd, 0, 0, 1.0f};
    gDisp("copy_nd", {d, x}, &pc, ceilDiv((uint64_t)hCnt * T * hd, 256), 1, 1);
}

static BufRef attnCore(const BufRef& q, const BufRef& k, const BufRef& v, uint32_t Tq, uint32_t Tk, uint32_t C,
                       uint32_t heads, bool causal = false) {
    uint32_t hd = C / heads;
    static int fusedEnv = -1;
    if (fusedEnv < 0) { const char* e = std::getenv("SDVK_ATTN_FUSED"); fusedEnv = (e && e[0] == '1') ? 1 : 0; }
    if (hd <= 64 && (hd == 64 || fusedEnv)) {
        BufRef out = allocN((uint64_t)Tq * C);
        if (!gArenaPlan) {
            ofc::gpu::FusedAttnArgs a{q, k, v, out, Tq, Tk, C, heads, causal};
            ofc::gpu::fusedAttention(G.eng, a);
        }
        return out;
    }
    uint64_t perHead = (uint64_t)Tq * Tk;
    uint32_t chunk = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(16, 16000000ull / perHead));
    // A single head over a large token count does not fit either (the VAE mid-block attends over
    // every latent pixel: 13824^2 floats is 764 MB). Cut the query axis too so the score tile stays
    // bounded, which makes the cost O(Tq*Tk) time but O(tile*Tk) memory.
    uint32_t qTile = Tq;
    if (perHead * chunk > 16000000ull) {
        qTile = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(Tq, 16000000ull / (Tk * chunk)));
        perHead = (uint64_t)qTile * Tk;
        ensureAttnScratch((uint64_t)chunk * perHead);
    } else {
        ensureAttnScratch((uint64_t)chunk * perHead);
    }
    BufRef out = allocN((uint64_t)Tq * C);
    float scale = 1.0f / std::sqrt((float)hd);
    for (uint32_t h0 = 0; h0 < heads; h0 += chunk) {
        uint32_t hc = std::min(chunk, heads - h0);
        BufRef qh = headsView(q, Tq, C, heads, h0, hc, scale);
        BufRef kh = headsView(k, Tk, C, heads, h0, hc, 1.0f);
        BufRef vh = headsView(v, Tk, C, heads, h0, hc, 1.0f);
        for (uint32_t q0 = 0; q0 < Tq; q0 += qTile) {
            uint32_t qc = std::min(qTile, Tq - q0);
            BufRef qs{qh.buf, qh.offset + (uint64_t)q0 * hd * 4, qh.size - (uint64_t)q0 * hd * 4};
            BufRef S{gAttnS.buf, gAttnS.offset, (uint64_t)hc * qc * Tk * 4};
            Pg pc{qc, Tk, hd, 0, Tq * hd, Tk * hd, qc * Tk, hc, 0, 0, 0, 0};
            gDisp("gemm_nt", {qs, kh, S}, &pc, ceilDiv(Tk, 64), ceilDiv(qc, 64), hc);
            if (causal) {
                struct {
                    uint32_t batch, rows, cols;
                } cm{hc, qc, Tk};
                gDisp("causal_mask", {S}, &cm, ceilDiv((uint64_t)hc * qc * Tk, 256), 1, 1);
            }
            P2 ps{hc * qc, Tk};
            gDisp("softmax_fwd", {S, S}, &ps, hc * qc, 1, 1);
            BufRef O = allocN((uint64_t)hc * qc * hd);
            Pg po{qc, hd, Tk, 0, qc * Tk, Tk * hd, qc * hd, hc, 0, 0, 0, 0};
            gDisp("gemm_nn", {S, vh, O}, &po, ceilDiv(hd, 64), ceilDiv(qc, 64), hc);
            BufRef Os{out.buf, out.offset + (uint64_t)q0 * C * 4, out.size - (uint64_t)q0 * C * 4};
            headsMergeInto(Os, O, qc, C, heads, h0, hc);
        }
    }
    return out;
}

static BufRef crossAttention(const BufRef& xq, const BufRef& xkv, const std::string& p, uint32_t Tq, uint32_t Tk,
                             uint32_t C, uint32_t heads) {
    BufRef q = linear(xq, p + ".to_q", Tq);
    BufRef k = linear(xkv, p + ".to_k", Tk);
    BufRef v = linear(xkv, p + ".to_v", Tk);
    BufRef o = attnCore(q, k, v, Tq, Tk, C, heads);
    return linear(o, p + ".to_out.0", Tq);
}


// ================================================================ CLIP BPE tokenizer

struct TokSet {
    std::map<std::string, int> vocab;
    std::map<std::pair<std::string, std::string>, int> merges;
    std::unordered_map<uint32_t, uint8_t> charByte;
};

static void utf8Append(std::string& s, uint32_t cp) {
    if (cp < 0x80) s.push_back((char)cp);
    else if (cp < 0x800) {
        s.push_back((char)(0xC0 | (cp >> 6)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back((char)(0xE0 | (cp >> 12)));
        s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        s.push_back((char)(0xF0 | (cp >> 18)));
        s.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

static std::vector<uint32_t> utf8Decode(const std::string& s, size_t& i) {
    std::vector<uint32_t> out;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        size_t n;
        if (c < 0x80) {
            cp = c;
            n = 1;
        } else if ((c >> 5) == 0x6) {
            cp = c & 0x1F;
            n = 2;
        } else if ((c >> 4) == 0xE) {
            cp = c & 0x0F;
            n = 3;
        } else {
            cp = c & 0x07;
            n = 4;
        }
        if (i + n > s.size()) break;
        for (size_t k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

static void buildByteMap(TokSet& ts) {
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++) ts.charByte[(uint32_t)cs[i]] = (uint8_t)bs[i];
}

static std::string decodeToken(const TokSet& ts, const std::string& enc) {
    std::string raw;
    size_t i = 0;
    auto cps = utf8Decode(enc, i);
    for (uint32_t cp : cps) {
        auto it = ts.charByte.find(cp);
        if (it == ts.charByte.end()) continue;
        raw.push_back((char)it->second);
    }
    return raw;
}

static void loadVocab(TokSet& ts, const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::printf("FAIL: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t i = 0;
    auto ws = [&]() {
        while (i < s.size() && std::isspace((unsigned char)s[i])) i++;
    };
    auto readStr = [&]() {
        std::string out;
        if (i >= s.size() || s[i] != '"') {
            std::printf("vocab: bad json\n");
            std::exit(1);
        }
        i++;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[i + 1];
                if (c == 'n') {
                    out.push_back('\n');
                    i += 2;
                } else if (c == 't') {
                    out.push_back('\t');
                    i += 2;
                } else if (c == 'r') {
                    out.push_back('\r');
                    i += 2;
                } else if (c == 'u' && i + 5 < s.size()) {
                    uint32_t cp = (uint32_t)std::stoul(s.substr(i + 2, 4), nullptr, 16);
                    i += 6;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 5 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        uint32_t lo = (uint32_t)std::stoul(s.substr(i + 2, 4), nullptr, 16);
                        i += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8Append(out, cp);
                } else {
                    out.push_back(c);
                    i += 2;
                }
            } else out.push_back(s[i++]);
        }
        i++;
        return out;
    };
    ws();
    if (i >= s.size() || s[i] != '{') {
        std::printf("vocab: expected object\n");
        std::exit(1);
    }
    i++;
    while (i < s.size()) {
        ws();
        if (s[i] == '}') break;
        if (s[i] == ',') {
            i++;
            continue;
        }
        std::string key = readStr();
        ws();
        if (i < s.size() && s[i] == ':') i++;
        ws();
        char* endp = nullptr;
        long id = std::strtol(s.c_str() + i, &endp, 10);
        i = (size_t)(endp - s.c_str());
        ts.vocab[decodeToken(ts, key)] = (int)id;
    }
}

static void loadMerges(TokSet& ts, const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::printf("FAIL: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = decodeToken(ts, line.substr(0, sp));
        std::string b = decodeToken(ts, line.substr(sp + 1));
        ts.merges[{a, b}] = rank++;
    }
}

static std::vector<int> bpeWord(const TokSet& ts, const std::string& word) {
    if (word.empty()) return {};
    std::vector<std::string> syms;
    for (unsigned char c : word) syms.push_back(std::string(1, (char)c));
    syms.back() += "</w>";  // CLIP bpe(): word = tuple(token[:-1]) + (token[-1] + "</w>",)
    while (syms.size() > 1) {
        int bestRank = INT32_MAX;
        size_t bestAt = 0;
        for (size_t i = 0; i + 1 < syms.size(); i++) {
            auto it = ts.merges.find({syms[i], syms[i + 1]});
            if (it != ts.merges.end() && it->second < bestRank) {
                bestRank = it->second;
                bestAt = i;
            }
        }
        if (bestRank == INT32_MAX) break;
        syms[bestAt] += syms[bestAt + 1];
        syms.erase(syms.begin() + bestAt + 1);
    }
    std::vector<int> ids;
    for (auto& s : syms) {
        auto it = ts.vocab.find(s);
        if (it != ts.vocab.end()) ids.push_back(it->second);
    }
    return ids;
}

static uint32_t utf8One(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        i++;
        return c;
    }
    uint32_t cp;
    size_t n;
    if ((c >> 5) == 0x6) {
        cp = c & 0x1F;
        n = 2;
    } else if ((c >> 4) == 0xE) {
        cp = c & 0x0F;
        n = 3;
    } else {
        cp = c & 0x07;
        n = 4;
    }
    if (i + n > s.size()) {
        i++;
        return 0;
    }
    for (size_t k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    i += n;
    return cp;
}

static std::vector<std::string> preTokenize(const std::string& text) {
    // CLIP pattern: 's|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+  (case-insensitive)
    std::string low;
    for (char c : text) low.push_back((char)std::tolower((unsigned char)c));
    std::vector<std::string> words;
    static const char* sufs[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    size_t i = 0;
    while (i < low.size()) {
        unsigned char c = (unsigned char)low[i];
        if (std::isspace(c)) {
            i++;
            continue;
        }
        if (c == '\'' && i + 1 < low.size()) {
            bool matched = false;
            for (const char* suf : sufs) {
                size_t sl = std::strlen(suf);
                if (low.compare(i + 1, sl, suf) == 0) {
                    words.push_back("'" + std::string(suf));
                    i += 1 + sl;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        if (c < 0x80 && std::isalpha(c)) {
            std::string w;
            while (i < low.size() && std::isalpha((unsigned char)low[i])) {
                w.push_back(low[i]);
                i++;
            }
            words.push_back(w);
            continue;
        }
        if (c < 0x80 && std::isdigit(c)) {
            std::string w;
            while (i < low.size() && std::isdigit((unsigned char)low[i])) {
                w.push_back(low[i]);
                i++;
            }
            words.push_back(w);
            continue;
        }
        if (c < 0x80) {
            std::string w;
            while (i < low.size()) {
                unsigned char d = (unsigned char)low[i];
                if (d >= 0x80 || std::isalnum(d) || std::isspace(d)) break;
                w.push_back((char)d);
                i++;
            }
            words.push_back(w);
            continue;
        }
        // non-ASCII: per character (keeps CJK prompts usable)
        uint32_t cp = utf8One(low, i);
        std::string one;
        utf8Append(one, cp);
        words.push_back(one);
    }
    return words;
}

static std::vector<int> clipTokenize(const TokSet& ts, const std::string& text) {
    std::vector<int> ids;
    ids.push_back(49406);
    for (auto& w : preTokenize(text)) {
        auto t = bpeWord(ts, w);
        for (int id : t) ids.push_back(id);
        if (ids.size() >= 76) break;
    }
    if (ids.size() > 75) ids.resize(75);
    ids.push_back(49407);
    while (ids.size() < 77) ids.push_back(49407);
    return ids;
}

// ================================================================ CLIP text encoder

// pfx covers "<tag>.text_model."; proj is "<tag>.text_projection" (empty = none)
// idx: device buffer with the 77 token ids (uploaded by the caller outside the step);
// embOut/pooledOut: device buffers written here, downloaded by the caller after waitStep
static void clipEncode(const std::string& pfx, const std::string& proj, const BufRef& idx, const std::vector<int>& ids,
                       uint32_t D, uint32_t layers, uint32_t heads, uint32_t actMode, const BufRef& embOut,
                       const BufRef& pooledOut) {
    const uint32_t T = 77;
    uint32_t ffn = wdimT(pfx + "encoder.layers.0.mlp.fc1.weight");

    BufRef xSlot = allocN((uint64_t)T * D);
    {
        BufRef tok = allocN((uint64_t)T * D);
        P2 pg{T, D};
        gDisp("gather_rows", {W(pfx + "embeddings.token_embedding.weight"), idx, tok}, &pg,
                       ceilDiv((uint64_t)T * D, 256), 1, 1);
        BufRef x0 = add2(tok, W(pfx + "embeddings.position_embedding.weight"), (uint64_t)T * D);
        if (gTrace && pfx == "te1.text_model.") tr("emb", x0, (uint64_t)T * D);
        copyInto(xSlot, 0, x0, (uint64_t)T * D * 4);
        arenaRewind(xSlot.offset + xSlot.size);
    }

    for (uint32_t l = 0; l < layers; l++) {
        std::string lp = pfx + "encoder.layers." + std::to_string(l) + ".";
        std::string tp = (pfx == "te1.text_model.") ? "t1_" : "t2_";
        bool tl = gTrace && l == 0;
        BufRef h = layernorm(xSlot, lp + "layer_norm1", T, D);
        if (tl) tr(tp + "ln1", h, (uint64_t)T * D);
        BufRef q = linear(h, lp + "self_attn.q_proj", T);
        if (tl) tr(tp + "q", q, (uint64_t)T * D);
        BufRef k = linear(h, lp + "self_attn.k_proj", T);
        BufRef v = linear(h, lp + "self_attn.v_proj", T);
        BufRef o = attnCore(q, k, v, T, T, D, heads, true);
        if (tl) tr(tp + "attn", o, (uint64_t)T * D);
        o = linear(o, lp + "self_attn.out_proj", T);
        if (tl) tr(tp + "oproj", o, (uint64_t)T * D);
        BufRef x1 = add2(xSlot, o, (uint64_t)T * D);
        if (tl) tr(tp + "x1", x1, (uint64_t)T * D);
        h = layernorm(x1, lp + "layer_norm2", T, D);
        if (tl) tr(tp + "ln2", h, (uint64_t)T * D);
        h = linear(h, lp + "mlp.fc1", T);
        if (tl) tr(tp + "fc1", h, (uint64_t)T * ffn);
        h = unary(h, (uint64_t)T * ffn, actMode);
        if (tl) tr(tp + "act", h, (uint64_t)T * ffn);
        h = linear(h, lp + "mlp.fc2", T);
        BufRef x2 = add2(x1, h, (uint64_t)T * D);
        if (tl) tr(tp + "x2", x2, (uint64_t)T * D);
        if (l + 2 == layers) copyInto(embOut, 0, x2, (uint64_t)T * D * 4);
        copyInto(xSlot, 0, x2, (uint64_t)T * D * 4);
        arenaRewind(xSlot.offset + xSlot.size);
    }

    bool needPool = !proj.empty() && hasW(proj + ".weight");
    if (proj.empty() && hasW(pfx + "final_layer_norm.weight")) needPool = true;
    if (needPool) {
        BufRef f = layernorm(xSlot, pfx + "final_layer_norm", T, D);
        uint32_t am = 0;
        for (uint32_t i = 1; i < T; i++)
            if (ids[i] > ids[am]) am = i;
        BufRef row{f.buf, f.offset + (uint64_t)am * D * 4, (uint64_t)D * 4};
        // CLIP-L has no projection head: its pooled embedding is the EOS hidden state as-is, so
        // only run the matmul when a projection weight actually exists
        BufRef p = row;
        if (hasW(proj.empty() ? pfx + "text_projection" : proj))
            p = linear(row, proj.empty() ? pfx + "text_projection" : proj, 1);
        copyInto(pooledOut, 0, p, (uint64_t)D * 4);
        arenaRewind(xSlot.offset + xSlot.size);
    }
}

// ================================================================ SDXL UNet

static BufRef resBlock(const BufRef& x, const std::string& p, const BufRef& temb, uint32_t H, uint32_t Wd,
                       const char* tag = "") {
    // conv weight dims are (cout, cin*k)
    uint32_t cout, kCols;
    linearDims(p + ".in_layers.2.weight", cout, kCols);
    uint32_t cin = kCols / 9;
    uint32_t P = H * Wd;
    bool sel = gTrace && *gDbSel && tag && *tag && strstr(gDbSel, tag) != nullptr;
    auto dt = [&](const char* suf, const BufRef& b, uint64_t floats) {
        if (sel) tr(std::string(tag) + suf, b, floats);
    };
    BufRef out = allocN((uint64_t)cout * P);
    BufRef h = gnSilu(x, p + ".in_layers.0", cin, P, 32, 1e-5f);
    dt("_n1", h, (uint64_t)cin * P);
    h = conv3Raw(h, p + ".in_layers.2.weight", cin, cout, H, Wd);
    dt("_c1", h, (uint64_t)cout * P);
    // diffusers ResnetBlock2D: temb = time_emb_proj(silu(temb)) before it is added to conv1
    uint32_t ecout, etembDim;
    linearDims(p + ".emb_layers.1.weight", ecout, etembDim);
    BufRef e = linear(silu(temb, etembDim), p + ".emb_layers.1", 1);
    dt("_e", e, cout);
    biasAddCh(h, e, cout, P);
    gBar();
    arenaRewind(h.offset);
    h = gnSilu(h, p + ".out_layers.0", cout, P, 32, 1e-5f);
    dt("_n2", h, (uint64_t)cout * P);
    h = conv3Raw(h, p + ".out_layers.3.weight", cout, cout, H, Wd);
    dt("_c2", h, (uint64_t)cout * P);
    BufRef sc = x;
    if (cin != cout) {
        sc = conv1(x, p + ".skip_connection", cin, cout, P);
        dt("_sc", sc, (uint64_t)cout * P);
    }
    BufRef r = add2(sc, h, (uint64_t)cout * P);
    copyInto(out, 0, r, (uint64_t)cout * P * 4);
    gBar();
    arenaRewind(out.offset + out.size);
    return out;
}

// diffusers GEGLU: y = gelu(secondHalf) * firstHalf, halves of the nout-wide linear output
static BufRef geglu(const BufRef& x, const std::string& p, uint32_t rows) {
    uint32_t nout, nin;
    linearDims(p + ".weight", nout, nin);
    uint32_t F = nout / 2;
    BufRef h = linear(x, p, rows);
    BufRef a = copyNd(h, rows * F, F, rows, 1, 1, F, 0, 1, nout, 0);
    BufRef src{h.buf, h.offset + (uint64_t)F * 4, h.size - (uint64_t)F * 4};
    BufRef b = copyNd(src, rows * F, F, rows, 1, 1, F, 0, 1, nout, 0);
    b = geluErf(b, (uint64_t)rows * F);
    return mul2(a, b, (uint64_t)rows * F);
}

static BufRef spatialTransformer(const BufRef& x, const std::string& p, const BufRef& ctx, uint32_t H, uint32_t Wd,
                                 uint32_t layers, const char* tag = "", uint32_t headsOverride = 0) {
    uint32_t C, tmp;
    linearDims(p + ".norm.weight", C, tmp);
    uint32_t P = H * Wd;
    uint32_t heads = headsOverride ? headsOverride : C / 64;
    bool sel = gTrace && *gDbSel && tag && *tag && strstr(gDbSel, tag) != nullptr;
    auto dt = [&](const char* suf, const BufRef& b, uint64_t floats) {
        if (sel) tr(std::string(tag) + suf, b, floats);
    };
    BufRef outSlot = allocN((uint64_t)C * P);
    BufRef h = gn(x, p + ".norm", C, P, 32, 1e-6f);
    dt("_gn", h, (uint64_t)C * P);
    h = conv1(h, p + ".proj_in", C, C, P);
    dt("_p", h, (uint64_t)C * P);
    BufRef t = transpose2(h, C, P);
    dt("_t", t, (uint64_t)P * C);
    for (uint32_t i = 0; i < layers; i++) {
        std::string tp = p + ".transformer_blocks." + std::to_string(i);
        bool l0 = sel && i == 0;
        BufRef n = layernorm(t, tp + ".norm1", P, C);
        if (l0) dt("_ln1", n, (uint64_t)P * C);
        BufRef a = crossAttention(n, n, tp + ".attn1", P, P, C, heads);
        if (l0) dt("_a1", a, (uint64_t)P * C);
        { P2 pa{(uint32_t)((uint64_t)P * C), 0}; gDisp("add2", {t, t, a}, &pa, ceilDiv((uint64_t)P * C, 256), 1, 1); }
        n = layernorm(t, tp + ".norm2", P, C);
        if (l0) dt("_ln2", n, (uint64_t)P * C);
        a = crossAttention(n, ctx, tp + ".attn2", P, 77, C, heads);
        if (l0) dt("_a2", a, (uint64_t)P * C);
        { P2 pa{(uint32_t)((uint64_t)P * C), 0}; gDisp("add2", {t, t, a}, &pa, ceilDiv((uint64_t)P * C, 256), 1, 1); }
        n = layernorm(t, tp + ".norm3", P, C);
        if (l0) dt("_ln3", n, (uint64_t)P * C);
        BufRef f = geglu(n, tp + ".ff.net.0.proj", P);
        if (l0) {
            uint32_t r0, c0;
            linearDims(tp + ".ff.net.0.proj.weight", r0, c0);
            dt("_g", f, (uint64_t)P * (r0 / 2));
        }
        f = linear(f, tp + ".ff.net.2", P);
        if (l0) dt("_f", f, (uint64_t)P * C);
        { P2 pa{(uint32_t)((uint64_t)P * C), 0}; gDisp("add2", {t, t, f}, &pa, ceilDiv((uint64_t)P * C, 256), 1, 1); }
    }
    BufRef o = transpose2(t, P, C);
    gBar();
    // The projection must not land on transposed t: an in-place gemm reads the rows it later
    // overwrites. Keep the transpose and free only the scratch above it.
    arenaRewind(o.offset + o.size);
    o = conv1(o, p + ".proj_out", C, C, P);
    dt("_po", o, (uint64_t)C * P);
    // only the slot survives, so the whole post-attention chain is dead once proj_out is done
    BufRef r = add2(x, o, (uint64_t)C * P);
    copyInto(outSlot, 0, r, (uint64_t)C * P * 4);
    gBar();
    arenaRewind(outSlot.offset + outSlot.size);
    return outSlot;
}

// sinusoidal embedding (diffusers get_timestep_embedding, flip_sin_to_cos=True, shift=0)
static std::vector<float> timeEmbHost(float t, uint32_t dim) {
    std::vector<float> e(dim);
    uint32_t half = dim / 2;
    for (uint32_t i = 0; i < half; i++) {
        float f = std::exp(-std::log(10000.0f) * (float)i / (float)half);
        e[i] = std::cos(t * f);
        e[half + i] = std::sin(t * f);
    }
    return e;
}

static BufRef concatCh(const BufRef& a, uint32_t ca, const BufRef& b, uint32_t cb, uint32_t P) {
    BufRef y = allocN((uint64_t)(ca + cb) * P);
    copyInto(y, 0, a, (uint64_t)ca * P * 4);
    copyInto(y, (uint64_t)ca * P * 4, b, (uint64_t)cb * P * 4);
    return y;
}

// SD1.5 (LDM) UNet: same building blocks as the SDXL path, different block order, attention
// placement and head width. Layout taken from the checkpoint: conv_in; [res+attn]x2, down,
// [res+attn]x2, down, [res+attn]x2, down, [res]x2; mid res+attn+res; then 12 up blocks that
// concat one skip, run a resnet, optionally attend (out.3..11) and optionally upsample
// (out.2 / out.5 / out.8, where the checkpoint carries a .1.conv / .2.conv).
static BufRef unetForwardSd15(const BufRef& lat, const BufRef& temb, const BufRef& ctx, uint32_t H, uint32_t Wd) {
    const uint32_t kHeads = 8;  // SD1.5 attention heads (head dim = C/8)
    std::vector<BufRef> hs;
    std::vector<uint32_t> hch;
    uint32_t h = H, w = Wd;
    BufRef x = conv3Raw(lat, "input_blocks.0.0.weight", 4, 320, H, Wd);
    uint32_t ch = 320;
    hs.push_back(x);
    hch.push_back(ch);
    auto down = [&](const char* pre, bool attn_) {
        std::string p = pre;
        // LDM checkpoints keep the resnet under an explicit .0 segment (SDXL conversions flatten it)
        x = resBlock(x, p + ".0", temb, h, w);
        ch = wdimT(p + ".0.out_layers.3.weight");
        if (attn_) x = spatialTransformer(x, p + ".1", ctx, h, w, 1, nullptr, kHeads);
        hs.push_back(x);
        hch.push_back(ch);
    };
    auto downSample = [&](const char* op) {
        x = conv3s2(x, op, h, w);
        ch = wdimT(std::string(op) + ".weight");
        h /= 2;
        w /= 2;
        hs.push_back(x);
        hch.push_back(ch);
    };
    down("input_blocks.1", true);
    down("input_blocks.2", true);
    downSample("input_blocks.3.0.op");
    down("input_blocks.4", true);
    down("input_blocks.5", true);
    downSample("input_blocks.6.0.op");
    down("input_blocks.7", true);
    down("input_blocks.8", true);
    downSample("input_blocks.9.0.op");
    down("input_blocks.10", false);
    down("input_blocks.11", false);

    x = resBlock(x, "middle_block.0", temb, h, w);
    ch = wdimT("middle_block.0.out_layers.3.weight");
    x = spatialTransformer(x, "middle_block.1", ctx, h, w, 1, nullptr, kHeads);
    x = resBlock(x, "middle_block.2", temb, h, w);
    ch = wdimT("middle_block.2.out_layers.3.weight");

    for (int j = 0; j < 12; j++) {
        std::string pre = "output_blocks." + std::to_string(j);
        uint32_t cs = hch.back();
        BufRef skip = hs.back();
        hs.pop_back();
        hch.pop_back();
        x = concatCh(x, ch, skip, cs, h * w);
        ch += cs;
        x = resBlock(x, pre + ".0", temb, h, w);
        ch = wdimT(pre + ".0.out_layers.3.weight");
        if (hasAnyW(pre + ".1.transformer_blocks.0.attn1.to_q.weight")) {
            x = spatialTransformer(x, pre + ".1", ctx, h, w, 1, nullptr, kHeads);
        }
        std::string upPfx;
        if (hasAnyW(pre + ".2.conv.weight")) upPfx = ".2.conv";
        else if (hasAnyW(pre + ".1.conv.weight")) upPfx = ".1.conv";
        if (!upPfx.empty()) {
            x = up2(x, ch, h, w);
            h *= 2;
            w *= 2;
            x = conv3Raw(x, pre + upPfx + ".weight", ch, ch, h, w);
        }
    }
    x = gnSilu(x, "out.0", ch, h * w, 32, 1e-5f);
    x = conv3Raw(x, "out.2.weight", ch, 4, h, w);
    return x;
}

static BufRef unetForward(const BufRef& lat, const BufRef& temb, const BufRef& ctx, uint32_t H, uint32_t Wd) {
    if (gArch == 1) return unetForwardSd15(lat, temb, ctx, H, Wd);
    // stage tags mirror ref/unet_stages.py so dbg_<tag>.f32 files line up 1:1
    static const char* kDnRes[6] = {"d0_r0", "d0_r1", "d1_r0", "d1_r1", "d2_r0", "d2_r1"};
    static const char* kDnAttn[6] = {"", "", "d1_a0", "d1_a1", "d2_a0", "d2_a1"};
    static const char* kUpRes[9] = {"u0_r0", "u0_r1", "u0_r2", "u1_r0", "u1_r1", "u1_r2", "u2_r0", "u2_r1", "u2_r2"};
    static const char* kUpAttn[9] = {"u0_a0", "u0_a1", "u0_a2", "u1_a0", "u1_a1", "u1_a2", "", "", ""};
    auto dbg = [&](const char* tag, const BufRef& b, uint32_t c, uint32_t hh, uint32_t ww) {
        if (gTrace && tag[0]) tr(tag, b, (uint64_t)c * hh * ww);
    };
    std::vector<BufRef> hs;
    std::vector<uint32_t> hch;
    uint32_t h = H, w = Wd;
    if (gArenaPlan) {
        planReset();
        std::printf("plan: canvas %ux%u latent %ux%u\n", w * 8, h * 8, w, h);
    }
    // the pass starts wherever earlier buffers left the cursor, so a plan number alone understates
    // what the pool has to hold
    if (gArenaPlan) std::printf("plan: base %.2f MB\n", gPlanUsed / 1048576.0);
    BufRef x = conv3Raw(lat, "input_blocks.0.0.weight", 4, 320, H, Wd);
    uint32_t ch = 320;
    dbg("conv_in", x, ch, h, w);
    hs.push_back(x);
    hch.push_back(ch);

    auto downModule = [&](uint32_t i, const std::string& pre, uint32_t h, uint32_t w, uint32_t tlayers) {
        x = resBlock(x, pre + ".0", temb, h, w, kDnRes[i]);
        ch = wdimT(pre + ".0.out_layers.3.weight");
        dbg(kDnRes[i], x, ch, h, w);
        if (tlayers) x = spatialTransformer(x, pre + ".1", ctx, h, w, tlayers, kDnAttn[i]);
        dbg(kDnAttn[i], x, ch, h, w);
        hs.push_back(x);
        hch.push_back(ch);
    };
    downModule(0, "input_blocks.1", h, w, 0);
    downModule(1, "input_blocks.2", h, w, 0);
    x = conv3s2(x, "input_blocks.3.0.op", h, w);
    ch = wdimT("input_blocks.3.0.op.weight");
    h /= 2;
    w /= 2;
    dbg("d0_ds", x, ch, h, w);
    hs.push_back(x);
    hch.push_back(ch);

    downModule(2, "input_blocks.4", h, w, 2);
    downModule(3, "input_blocks.5", h, w, 2);
    x = conv3s2(x, "input_blocks.6.0.op", h, w);
    ch = wdimT("input_blocks.6.0.op.weight");
    h /= 2;
    w /= 2;
    dbg("d1_ds", x, ch, h, w);
    hs.push_back(x);
    hch.push_back(ch);

    downModule(4, "input_blocks.7", h, w, 10);
    downModule(5, "input_blocks.8", h, w, 10);

    x = resBlock(x, "middle_block.0", temb, h, w, "m_r0");
    ch = wdimT("middle_block.0.out_layers.3.weight");
    dbg("m_r0", x, ch, h, w);
    x = spatialTransformer(x, "middle_block.1", ctx, h, w, 10, "m_a0");
    dbg("m_a0", x, ch, h, w);
    x = resBlock(x, "middle_block.2", temb, h, w, "m_r1");
    ch = wdimT("middle_block.2.out_layers.3.weight");
    dbg("m_r1", x, ch, h, w);

    for (int j = 0; j < 9; j++) {
        std::string pre = "output_blocks." + std::to_string(j);
        uint32_t cs = hch.back();
        BufRef skip = hs.back();
        hs.pop_back();
        hch.pop_back();
        x = concatCh(x, ch, skip, cs, h * w);
        ch = ch + cs;
        uint32_t tlayers = (j < 3) ? 10 : (j < 6) ? 2 : 0;
        x = resBlock(x, pre + ".0", temb, h, w, kUpRes[j]);
        ch = wdimT(pre + ".0.out_layers.3.weight");
        dbg(kUpRes[j], x, ch, h, w);
        if (tlayers) x = spatialTransformer(x, pre + ".1", ctx, h, w, tlayers, kUpAttn[j]);
        dbg(kUpAttn[j], x, ch, h, w);
        std::string upPfx;
        if (hasAnyW(pre + ".2.conv.weight")) upPfx = ".2.conv";
        else if (hasAnyW(pre + ".1.conv.weight")) upPfx = ".1.conv";
        if (!upPfx.empty()) {
            x = up2(x, ch, h, w);
            h *= 2;
            w *= 2;
            x = conv3Raw(x, pre + upPfx + ".weight", ch, ch, h, w);
            dbg(j < 3 ? "u0_up" : "u1_up", x, ch, h, w);
        }
    }

    x = gnSilu(x, "out.0", ch, h * w, 32, 1e-5f);
    dbg("conv_gn", x, ch, h, w);
    x = conv3Raw(x, "out.2.weight", ch, 4, h, w);
    dbg("conv_out", x, 4, h, w);
    if (gArenaPlan) {
        gArenaPlan = false;
        if (!gAutoPlan) {
            std::printf("plan: peak %.2f MB at canvas %ux%u\n", gPlanPeak / 1048576.0, w * 8, h * 8);
            std::exit(0);
        }
    }
    return x;
}

// temb = time_embed(get_timestep_embedding(t)) + add_embedding(concat(pooled, add_time_ids))
static BufRef buildTemb(const BufRef& pool, const BufRef& addTime1536, const BufRef& tEmb320) {
    BufRef t = linear(tEmb320, "time_embed.0", 1);
    t = silu(t, (uint64_t)wdimT("time_embed.0.weight"));
    t = linear(t, "time_embed.2", 1);
    if (gArch == 1) return t;  // SD1.5 has no add_embedding (pooled + size conditioning)
    BufRef cat = allocN(2816);
    copyInto(cat, 0, pool, 1280 * 4);
    copyInto(cat, 1280 * 4, addTime1536, 1536 * 4);
    BufRef a = linear(cat, "label_emb.0.0", 1);
    a = silu(a, (uint64_t)wdimT("label_emb.0.0.weight"));
    a = linear(a, "label_emb.0.2", 1);
    return add2(t, a, 1280);
}

// ================================================================ VAE decoder

// A crop-and-scale on load lets an img2img request seed the sampler without a separate encoder.
static BufRef vaeAttn(const BufRef& x, const std::string& p, uint32_t C, uint32_t H, uint32_t Wd) {
    BufRef h = gn(x, p + ".group_norm", C, H * Wd, 32, 1e-6f);
    // The conv chain is channel-major ([C, P]) but QKV projections and attnCore work on
    // token-major ([P, C]) rows; feeding the conv layout straight into linear() scrambles
    // channels and shows up as high-frequency speckle in the decoded image.
    BufRef ht = transpose2(h, C, H * Wd);
    BufRef q = linear(ht, p + ".to_q", H * Wd);
    BufRef k = linear(ht, p + ".to_k", H * Wd);
    BufRef v = linear(ht, p + ".to_v", H * Wd);
    uint32_t heads = 1;
    BufRef o = attnCore(q, k, v, H * Wd, H * Wd, C, heads, false);
    // Rewinding to o.offset here would make the projection below write its output straight onto
    // the gemm input; the caller's compact() already reclaims these temporaries.
    o = linear(o, p + ".to_out.0", H * Wd);
    o = transpose2(o, H * Wd, C);
    return add2(x, o, (uint64_t)C * H * Wd);
}

static BufRef vaeResnet(const BufRef& x, const std::string& p, uint32_t cin, uint32_t cout, uint32_t H, uint32_t Wd,
                        bool shortcut) {
    uint32_t P = H * Wd;
    uint64_t slotMark = arenaMark();
    // When the block narrows (512 -> 256) or keeps its width, the input's own span is big enough to
    // hold the result, so the block writes back into it. That removes one 452 MB tensor from the
    // peak at 1152x768, which is the difference between decoding and dying on the last up-block.
    bool reuse = !shortcut && x.size >= (uint64_t)cout * P * 4;
    BufRef slot = reuse ? BufRef{x.buf, x.offset, (uint64_t)cout * P * 4} : allocN((uint64_t)cout * P);
    // Two reusable scratch slots, alternating: the chain below is a straight line where each result
    // only needs to outlive the next op, but every op allocates its output, so a 128-channel block
    // at 1152x768 (452 MB a copy) used to stack six of them and blow the pool on the last block.
    uint64_t cw = (uint64_t)std::max(cin, cout) * P;
    BufRef a = allocN(cw);
    BufRef b = allocN(cw);
    // Ping-pong between a and b: a 3x3 conv stepped in place would clobber the rows the next row
    // chunk still has to read, so src and dst must never be the same span.
    static int vaeSubCalls = 0;
    ++vaeSubCalls;
    const char* subName = std::getenv("OFC_VAE_TRACE_RESNET");
    bool sub = gVaeTraceSub && subName && p.find(subName) != std::string::npos;
    auto subSnap = [&](const char* tag, const BufRef& buf, uint64_t floats) {
        if (!sub) return;
        tr(std::string("sub_") + tag, buf, floats);
    };
    BufRef h = gnInto(x, p + ".norm1", cin, P, 32, 1e-6f, a, 1);
    subSnap("gn1", h, (uint64_t)cin * P);
    h = conv3RawInto(h, p + ".conv1.weight", cin, cout, H, Wd, b);
    subSnap("c1", h, (uint64_t)cout * P);
    h = gnInto(h, p + ".norm2", cout, P, 32, 1e-6f, a, 1);
    subSnap("gn2", h, (uint64_t)cout * P);
    h = conv3RawInto(h, p + ".conv2.weight", cout, cout, H, Wd, b);
    subSnap("c2", h, (uint64_t)cout * P);
    BufRef sc = x;
    if (shortcut) sc = conv1(x, p + ".conv_shortcut", cin, cout, P);
    BufRef r = add2Into(sc, h, (uint64_t)cout * P, h);  // elementwise, in-place on h is safe
    // r aliases x when the result already landed in the recycle slots, so the copy out has to come
    // first and the source must not be the destination
    copyInto(slot, 0, r, (uint64_t)cout * P * 4);
    gBar();
    if (gArenaPlan) gPlanUsed = slotMark + (reuse ? 0 : slot.size);
    else G.eng.stepArena().rewind(reuse ? slotMark : slotMark + slot.size);
    return slot;
}

static BufRef vaeUpConv(const BufRef& x, const std::string& p, uint32_t C, uint32_t& H, uint32_t& Wd) {
    BufRef u = up2(x, C, H, Wd);
    H *= 2;
    Wd *= 2;
    return conv3Raw(u, p + ".weight", C, C, H, Wd);
}

// Full VAE decode pass: load VAE weights and decode a 4xHxWd latent (sampler space, i.e.
// still in the x0 scale) into CHW RGB in [-1,1].
static std::vector<float> vaeDecodeChain(const GenParams& p, const std::vector<float>& lat, uint32_t H,
                                         uint32_t Wd) {
    uint64_t latFloats = (uint64_t)4 * H * Wd;
    // weights are already resident: vaeDecode() loads them once for the full and the tiled paths
    // vaeResnet slots rewind after each block, so the step arena holds one live resnet chain,
    // not the sum of all activations; cap the estimate so 96^2 stays under the 6 GiB ceiling
    // Measured need: ~2.65 GB at a 96x96 latent, ~3.25 GB at 96x128 (the old 2600/4096 slope
    // plus a 2400 cap was short on non-square canvases). Scale by latent area with margin and
    // keep an explicit --vstep-mb as the floor; OFC_NO_AUTOSIZE restores the caller's value.
    uint64_t autoVStep = 400 + (uint64_t)H * Wd * 290 / 1024;
    if (autoVStep > 4000) autoVStep = 4000;
    if (std::getenv("OFC_NO_AUTOSIZE")) autoVStep = 0;
    uint64_t vStepMB = std::max<uint64_t>(p.vStepMb, autoVStep);
    G.eng.initArenas(vStepMB << 20, 220ull << 20);
    gResizeScratch(220ull << 20);
    gCol = G.scratchArena.alloc(kColFloats * 4);
    gSCap = 0;
    if (const char* env = std::getenv("OFC_ARENA_PLAN")) {
        if (env[0] && env[0] != '0') {
            gArenaPlan = true;
            planReset();
            std::printf("plan: vae %ux%u\n", H * 8, Wd * 8);
        }
    }
    auto vaeT0 = std::chrono::steady_clock::now();
    // OFC_VAE_TRACE=1 dumps every decoder stage (dbg_vae_*.f32) for numeric A/B against a
    // reference implementation; only sane on small latents (full tensors, scratch arena grows).
    bool vaeTrace = std::getenv("OFC_VAE_TRACE") != nullptr;
    if (vaeTrace) gTrace = true;
    gVaeTraceSub = std::getenv("OFC_VAE_TRACE_SUB") != nullptr;
    int snapNo = 0;

    G.eng.beginStep();
    std::vector<float> latScaled(latFloats);
    const float vaeScale = (gArch == 1) ? 0.18215f : 0.13025f;  // SD1.5 VAE scaling factor
    for (uint64_t k = 0; k < latFloats; k++) latScaled[k] = lat[k] / vaeScale;
    BufRef x = allocN(latFloats);
    auto snap = [&](const char* tag) {
        if (!vaeTrace) return;
        char nm[96];
        std::snprintf(nm, sizeof nm, "vae_%02d_%s", ++snapNo, tag);
        tr(nm, x, x.size / 4);
    };
    G.eng.uploadNow(x, latScaled.data(), latScaled.size() * 4);

    x = conv1(x, "post_quant_conv", 4, 4, H * Wd);
    snap("postq");
    // The decode chain is a straight line: each stage's result dies the moment the next stage
    // returns, but every allocator moves the cursor up one tensor. Rewinding the tail past the
    // live tensor caps the peak at two stages instead of the sum of all of them (1954 MB at
    // 768x1152) and, unlike a move-to-base, never submits work in the middle of recording.
    auto compact = [&]() {
        if (gArenaPlan) return;
        if (std::getenv("OFC_NO_COMPACT")) return;  // debug: isolate the in-stream move
        if (x.offset != 0) G.eng.compactStepArena(0, x.offset, x.size);
        x.offset = 0;
    };
    x = conv3Raw(x, "decoder.conv_in.weight", 4, 512, H, Wd);
    snap("conv_in");
    compact();
    x = vaeResnet(x, "decoder.mid_block.resnets.0", 512, 512, H, Wd, false);
    snap("mid_res0");
    compact();
    x = vaeAttn(x, "decoder.mid_block.attentions.0", 512, H, Wd);
    snap("mid_attn");
    compact();
    x = vaeResnet(x, "decoder.mid_block.resnets.1", 512, 512, H, Wd, false);
    snap("mid_res1");
    compact();
    char pfx[96];
    for (int b = 0; b < 4; b++) {
        uint32_t ci = (b <= 1) ? 512 : (b == 2 ? 512 : 256);
        uint32_t co = (b <= 1) ? 512 : (b == 2 ? 256 : 128);
        for (int i = 0; i < 3; i++) {
            std::snprintf(pfx, sizeof pfx, "decoder.up_blocks.%d.resnets.%d", b, i);
            x = vaeResnet(x, pfx, ci, co, H, Wd, i == 0 && ci != co);
            ci = co;  // resnets 1 and 2 read the narrowed width; passing the block's entry
                      // width again made them normalize/conv over channels that are not there
            snap(pfx);
            compact();
        }
        if (b < 3) {
            std::snprintf(pfx, sizeof pfx, "decoder.up_blocks.%d.upsamplers.0.conv", b);
            uint32_t C = (b == 2) ? 256 : 512;
            x = vaeUpConv(x, pfx, C, H, Wd);
            snap(pfx);
            compact();
        }
    }
    x = gnSilu(x, "decoder.conv_norm_out", 128, H * Wd, 32, 1e-6f);
    snap("norm_out");
    compact();
    x = conv3Raw(x, "decoder.conv_out.weight", 128, 3, H, Wd);
    snap("conv_out");
    if (gArenaPlan) {
        std::printf("plan: vae peak %.2f MB at %ux%u\n", gPlanPeak / 1048576.0, Wd * 8, H * 8);
        std::exit(0);
    }
    G.eng.endStep();
    G.eng.waitStep();
    if (vaeTrace) {
        flushTrace();
        gTrace = false;
    }
    // Row-chunked pull-back: the full fp32 image is 3*4*H*Wd bytes (~10 MB at 768x1152), but the
    // driver's readback staging buffer costs the same again, and on a 16 GB box that difference
    // decides whether the decode finishes. One slice at a time keeps the transient tiny.
    std::vector<float> img((size_t)3 * H * Wd);
    const uint32_t rowFloats = 3 * Wd;
    const uint32_t rowsPerChunk = std::max<uint32_t>(1, (1u << 20) / (rowFloats * 4));
    for (uint32_t y = 0; y < H; y += rowsPerChunk) {
        uint32_t rows = std::min(rowsPerChunk, H - y);
        G.eng.downloadNow(x, img.data() + (size_t)y * rowFloats, (uint64_t)rows * rowFloats * 4, (uint64_t)y * rowFloats * 4);
    }

    std::printf("vae decode %ux%u: %.0f ms (step arena %llu MB)\n", Wd, H,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vaeT0).count(),
                (unsigned long long)vStepMB);
    return img;
}

// Tiled decode: the full-image pass needs ~2.6-3.3 GB of device-local memory at 768-class
// canvases, which a 6 GB card cannot always give (the commit charge alone can refuse it).
// Splitting the latent into overlapping tiles keeps each pass at tile-sized memory; the overlap
// is blended with complementary linear ramps so the seams average out.
static std::vector<float> vaeDecodeTiled(const GenParams& p, const std::vector<float>& lat, uint32_t H, uint32_t Wd,
                                         uint32_t tile, uint32_t ov) {
    ov = std::max(1u, std::min(ov, tile / 2));
    const uint32_t step = tile - ov;
    const uint32_t OH = H * 8, OW = Wd * 8, ovPx = ov * 8;
    std::vector<float> acc((size_t)3 * OH * OW, 0.0f), wsum((size_t)OH * OW, 0.0f);
    uint32_t tiles = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t y0 = 0; y0 < H;) {
        // Align the last row band to the bottom edge instead of letting it shrink: a 96-high
        // latent cut into 48-px tiles would otherwise leave a 16-px band whose GroupNorm
        // statistics come from a sliver of the image and tint that strip.
        uint32_t yy = (H > tile) ? std::min(y0, H - tile) : 0u;
        uint32_t y1 = std::min(H, yy + tile);
        for (uint32_t x0 = 0; x0 < Wd;) {
            if (p.shouldCancel && p.shouldCancel()) fail("cancelled");
            uint32_t xx = (Wd > tile) ? std::min(x0, Wd - tile) : 0u;
            uint32_t x1 = std::min(Wd, xx + tile);
            uint32_t th = y1 - yy, tw = x1 - xx, th8 = th * 8, tw8 = tw * 8;
            std::vector<float> sub((size_t)4 * th * tw);
            for (uint32_t c = 0; c < 4; c++)
                for (uint32_t y = 0; y < th; y++)
                    std::memcpy(&sub[((size_t)c * th + y) * tw], &lat[((size_t)c * H + yy + y) * Wd + xx], tw * 4);
            std::vector<float> img = vaeDecodeChain(p, sub, th, tw);
            for (uint32_t c = 0; c < 3; c++)
                for (uint32_t y = 0; y < th8; y++) {
                    float wy = 1.0f;
                    if (yy > 0) wy = std::min(wy, (y + 0.5f) / (float)ovPx);
                    if (y1 < H) wy = std::min(wy, (th8 - y - 0.5f) / (float)ovPx);
                    for (uint32_t x = 0; x < tw8; x++) {
                        float wx = 1.0f;
                        if (xx > 0) wx = std::min(wx, (x + 0.5f) / (float)ovPx);
                        if (x1 < Wd) wx = std::min(wx, (tw8 - x - 0.5f) / (float)ovPx);
                        float w = wx * wy;
                        size_t gi = ((size_t)c * OH + (yy * 8 + y)) * OW + (xx * 8 + x);
                        acc[gi] += img[((size_t)c * th8 + y) * tw8 + x] * w;
                        if (c == 0) wsum[(size_t)(yy * 8 + y) * OW + (xx * 8 + x)] += w;
                    }
                }
            tiles++;
            if (x1 >= Wd) break;
            x0 = xx + step;
        }
        if (y1 >= H) break;
        y0 = yy + step;
    }
    // acc is planar [c][y][x], so the weight lookup must follow the same layout; indexing it as
    // wsum[i / 3] assumed an interleaved [y][x][c] buffer and normalized every pixel with the
    // weight of a different pixel, which showed up as horizontal scan lines in every tiled decode.
    for (uint32_t c = 0; c < 3; c++)
        for (uint32_t y = 0; y < OH; y++)
            for (uint32_t x = 0; x < OW; x++) {
                size_t i = ((size_t)c * OH + y) * OW + x;
                acc[i] /= std::max(wsum[(size_t)y * OW + x], 1e-6f);
            }
    std::printf("vae tiled decode %ux%u: %u tiles of %u px (+%u overlap), %.0f ms\n", OW, OH, tiles, tile, ov,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return acc;
}

// Bilinear latent resample for the hires refine stage (the VAE never sees this directly).
static std::vector<float> upscaleLatentBilinear(const std::vector<float>& in, uint32_t H, uint32_t W,
                                                uint32_t nh, uint32_t nw) {
    std::vector<float> out((size_t)4 * nh * nw);
    const float sy = (float)H / (float)nh, sx = (float)W / (float)nw;
    for (uint32_t c = 0; c < 4; c++) {
        const float* src = &in[(size_t)c * H * W];
        float* dst = &out[(size_t)c * nh * nw];
        for (uint32_t y = 0; y < nh; y++) {
            float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = (int)std::floor(fy);
            float ty = fy - (float)y0;
            int ya = std::min<int>(std::max(y0, 0), (int)H - 1);
            int yb = std::min<int>(std::max(y0 + 1, 0), (int)H - 1);
            for (uint32_t x = 0; x < nw; x++) {
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = (int)std::floor(fx);
                float tx = fx - (float)x0;
                int xa = std::min<int>(std::max(x0, 0), (int)W - 1);
                int xb = std::min<int>(std::max(x0 + 1, 0), (int)W - 1);
                float a = src[(size_t)ya * W + xa] * (1.0f - tx) + src[(size_t)ya * W + xb] * tx;
                float b = src[(size_t)yb * W + xa] * (1.0f - tx) + src[(size_t)yb * W + xb] * tx;
                dst[(size_t)y * nw + x] = a * (1.0f - ty) + b * ty;
            }
        }
    }
    return out;
}

// Bicubic (Catmull-Rom) RGB upscale on the decoded image. Upscaling the latent and running the
// VAE on it looks tempting but the decoder turns the interpolated latent into a hard 8-px grid,
// so the plain hires stage works on pixels instead.
static void upscaleImageBicubic(std::vector<float>& img, uint32_t& w, uint32_t& h, float f) {
    uint32_t nw = std::max(8u, (uint32_t)std::lround((double)w * f));
    uint32_t nh = std::max(8u, (uint32_t)std::lround((double)h * f));
    std::vector<float> out((size_t)3 * nw * nh);
    auto wk = [](float x) {
        float a = -0.5f, ax = std::fabs(x);
        if (ax <= 1.0f) return (a + 2.0f) * ax * ax * ax - (a + 3.0f) * ax * ax + 1.0f;
        if (ax < 2.0f) return a * ax * ax * ax - 5.0f * a * ax * ax + 8.0f * a * ax - 4.0f * a;
        return 0.0f;
    };
    const float sx = (float)w / (float)nw, sy = (float)h / (float)nh;
    for (uint32_t c = 0; c < 3; c++) {
        const float* src = &img[(size_t)c * w * h];
        float* dst = &out[(size_t)c * nw * nh];
        for (uint32_t y = 0; y < nh; y++) {
            float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = (int)std::floor(fy);
            float ty = fy - (float)y0;
            for (uint32_t x = 0; x < nw; x++) {
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = (int)std::floor(fx);
                float tx = fx - (float)x0;
                float acc = 0.0f, wsum = 0.0f;
                for (int j = -1; j <= 2; j++) {
                    float wy = wk((float)j - ty);
                    int yy = std::min<int>(std::max(y0 + j, 0), (int)h - 1);
                    for (int i = -1; i <= 2; i++) {
                        float wx = wk((float)i - tx);
                        int xx = std::min<int>(std::max(x0 + i, 0), (int)w - 1);
                        acc += src[(size_t)yy * w + xx] * wx * wy;
                        wsum += wx * wy;
                    }
                }
                dst[(size_t)y * nw + x] = acc / std::max(wsum, 1e-6f);
            }
        }
    }
    img.swap(out);
    std::printf("hires: %ux%u -> %ux%u (bicubic)\n", w, h, nw, nh);
    w = nw;
    h = nh;
}

// The VAE cannot live beside the UNet in the weight arena, so both the encode and the decode pass
// hand the text-encoder/UNet arena back before loading it.
static void prepareVaeArena(const GenParams& p) {
    clearWeights();
    // the sampler's step arena is still resident here; drop it first or the VAE cannot get room
    G.eng.initArenas(128ull << 20, 128ull << 20);
    G.paramArena.init(G.eng.device(), G.eng.physDevice(), 640ull << 20, true, false);
    loadVaeWeights(p);
}

static std::vector<float> vaeDecode(const GenParams& p, const std::vector<float>& lat, uint32_t H, uint32_t Wd) {
    prepareVaeArena(p);
    // Large canvases decode in tiles by default: a full 96x96 pass asks for ~3 GB in one
    // device-local allocation, and this driver kills the process outright (no error return)
    // when that allocation cannot be satisfied. Tiles keep every pass small; --vae-tile 0
    // forces the full-image pass back on.
    uint32_t tile = (H * Wd > 64u * 64u) ? p.vaeTile : 0;
    if (tile) return vaeDecodeTiled(p, lat, H, Wd, tile, p.vaeOverlap);
    try {
        return vaeDecodeChain(p, lat, H, Wd);
    } catch (const std::exception& e) {
        if (H * Wd <= 64 * 64 || std::getenv("OFC_NO_AUTOTILE")) throw;
        std::printf("vae: full decode failed (%s), retrying tiled\n", e.what());
        return vaeDecodeTiled(p, lat, H, Wd, 48, 8);
    }
}

// ================================================================ VAE encoder

// The decoder walked backwards: conv_in -> 4x (2 resnets [+ stride-2 downsample]) -> mid
// (resnet / attention / resnet) -> conv_norm_out -> conv_out (8 ch) -> quant_conv, and the first
// four channels of the result are the latent mean. Every VAE file carries these weights; the
// loader used to filter encoder.* out, which is why img2img could only seed the sampler with
// pixels * scaling_factor. It runs on the same ops as the decode path, in sampler scale.
static std::vector<float> vaeEncodeChain(const GenParams& p, const std::vector<float>& img, uint32_t H,
                                         uint32_t Wd) {
    if (H < 8 || Wd < 8) fail("init image too small: need at least 8x8 px after resampling");
    uint64_t srcFloats = (uint64_t)3 * H * Wd;
    if (img.size() < srcFloats) fail("vae encode: pixel buffer is smaller than the canvas");
    // The peak sits in down_blocks.0 at full resolution: the live tensor, its result slot and the
    // two ping-pong buffers the resnet needs, all 128 channels. Refuse up front rather than let
    // the driver kill the process on a canvas that cannot fit.
    uint64_t need = (uint64_t)4 * 128 * H * Wd * 4;
    uint64_t autoVStep = (need >> 20) * 115 / 100 + 96;
    if (std::getenv("OFC_NO_AUTOSIZE")) autoVStep = 0;
    uint64_t vStepMB = std::max<uint64_t>(p.vStepMb, autoVStep);
    if (vStepMB > 4200) {
        char msg[288];
        std::snprintf(msg, sizeof msg,
                      "init image at %ux%u needs ~%.0f MB of activations; use a canvas of 1280 px or less",
                      Wd, H, need / 1048576.0);
        fail(msg);
    }
    G.eng.initArenas(vStepMB << 20, 220ull << 20);
    gResizeScratch(220ull << 20);
    gCol = G.scratchArena.alloc(kColFloats * 4);
    gSCap = 0;
    auto t0 = std::chrono::steady_clock::now();
    G.eng.beginStep();
    BufRef x = allocN(srcFloats);
    G.eng.uploadNow(x, img.data(), srcFloats * 4);
    auto compact = [&]() {
        if (std::getenv("OFC_NO_COMPACT")) return;
        if (x.offset != 0) G.eng.compactStepArena(0, x.offset, x.size);
        x.offset = 0;
    };
    x = conv3Raw(x, "encoder.conv_in.weight", 3, 128, H, Wd);
    compact();
    uint32_t cin = 128;
    char pfx[96];
    for (int b = 0; b < 4; b++) {
        uint32_t cout = (b == 0) ? 128u : (b == 1 ? 256u : 512u);
        for (int i = 0; i < 2; i++) {
            std::snprintf(pfx, sizeof pfx, "encoder.down_blocks.%d.resnets.%d", b, i);
            x = vaeResnet(x, pfx, cin, cout, H, Wd, i == 0 && cin != cout);
            cin = cout;
            compact();
        }
        if (b < 3) {
            std::snprintf(pfx, sizeof pfx, "encoder.down_blocks.%d.downsamplers.0.conv", b);
            x = conv3s2(x, pfx, H, Wd, 0);  // diffusers pads right/bottom, not both sides
            H /= 2;
            Wd /= 2;
            compact();
        }
    }
    uint32_t P = H * Wd;
    x = vaeResnet(x, "encoder.mid_block.resnets.0", 512, 512, H, Wd, false);
    compact();
    x = vaeAttn(x, "encoder.mid_block.attentions.0", 512, H, Wd);
    compact();
    x = vaeResnet(x, "encoder.mid_block.resnets.1", 512, 512, H, Wd, false);
    compact();
    x = gnSilu(x, "encoder.conv_norm_out", 512, P, 32, 1e-6f);
    compact();
    x = conv3Raw(x, "encoder.conv_out.weight", 512, 8, H, Wd);
    compact();
    x = conv1(x, "quant_conv", 8, 8, P);
    G.eng.endStep();
    G.eng.waitStep();
    std::vector<float> raw((size_t)8 * P);
    G.eng.downloadNow(x, raw.data(), raw.size() * 4);
    // encoder output lives in VAE space; vaeDecodeChain divides the sampler latent by this same
    // factor, so multiply here and the two paths agree.
    const float vaeScale = (gArch == 1) ? 0.18215f : 0.13025f;
    std::vector<float> mean((size_t)4 * P);
    for (uint32_t c = 0; c < 4; c++)
        for (uint32_t i = 0; i < P; i++) mean[(size_t)c * P + i] = raw[(size_t)c * P + i] * vaeScale;
    std::printf("vae encode %ux%u -> %ux%u: %.0f ms (step arena %llu MB)\n", Wd * 8, H * 8, Wd, H,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                (unsigned long long)vStepMB);
    return mean;
}

// ================================================================ PNG writer

// zero-dependency PNG: zlib stream, fixed-Huffman LZ77 or stored blocks, optional tEXt chunk
static void pngChunk(FILE* f, const char* type, const uint8_t* data, uint32_t len) {
    static uint32_t tab[256];
    static bool init = false;
    if (!init) {
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            tab[n] = c;
        }
        init = true;
    }
    auto be32 = [](uint32_t v, uint8_t* p) {
        p[0] = (uint8_t)(v >> 24);
        p[1] = (uint8_t)(v >> 16);
        p[2] = (uint8_t)(v >> 8);
        p[3] = (uint8_t)v;
    };
    uint8_t hb[8];
    be32(len, hb);
    for (int i = 0; i < 4; i++) hb[4 + i] = (uint8_t)type[i];
    std::fwrite(hb, 1, 8, f);
    if (len) std::fwrite(data, 1, len, f);
    uint32_t crc = 0xFFFFFFFFu;
    for (int i = 0; i < 4; i++) crc = tab[(crc ^ (uint8_t)type[i]) & 0xFF] ^ (crc >> 8);
    for (uint32_t i = 0; i < len; i++) crc = tab[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    crc ^= 0xFFFFFFFFu;
    uint8_t cb[4];
    be32(crc, cb);
    std::fwrite(cb, 1, 4, f);
}

static std::vector<uint8_t> deflateFixed(const std::vector<uint8_t>& raw) {
    static const uint16_t lenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                         31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static const uint8_t lenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                         2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static const uint16_t distBase[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,
                                          33,  49,  65,  97,  129, 193,  257,  385,  513,  769,
                                          1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static const uint8_t distExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                                          6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    uint64_t bitBuf = 0;
    int bitCnt = 0;
    std::vector<uint8_t> body;
    body.reserve(raw.size() / 2 + 16);
    auto putBits = [&](uint32_t v, int n) {
        bitBuf |= (uint64_t)v << bitCnt;
        bitCnt += n;
        while (bitCnt >= 8) {
            body.push_back((uint8_t)bitBuf);
            bitBuf >>= 8;
            bitCnt -= 8;
        }
    };
    auto putCode = [&](uint32_t code, int n) {  // Huffman codes go MSB-first
        for (int i = n - 1; i >= 0; i--) putBits((code >> i) & 1u, 1);
    };
    putBits(1, 1);  // BFINAL
    putBits(1, 2);  // BTYPE = fixed Huffman
    const size_t kWin = 32768, kMax = 258, kHashBits = 15;
    std::vector<int32_t> head((size_t)1 << kHashBits, -1);
    std::vector<int32_t> prev(kWin, -1);
    auto hash3 = [&](size_t i) {
        uint32_t h =
            (uint32_t)raw[i] * 0x9E3779B1u ^ (uint32_t)raw[i + 1] * 0x85EBCA6Bu ^ (uint32_t)raw[i + 2] * 0xC2B2AE35u;
        return (h >> (32 - kHashBits)) & (((size_t)1 << kHashBits) - 1);
    };
    auto insert = [&](size_t pos) {
        if (pos + 2 >= raw.size()) return;
        size_t h = hash3(pos);
        prev[pos & (kWin - 1)] = head[h];
        head[h] = (int32_t)pos;
    };
    size_t i = 0, n = raw.size();
    while (i < n) {
        size_t bestLen = 0, bestDist = 0;
        if (i + 2 < n) {
            int32_t cand = head[hash3(i)];
            for (int chain = 0; cand >= 0 && chain < 32; chain++) {
                size_t d = i - (size_t)cand;
                if (d == 0 || d > kWin || cand >= (int32_t)i) break;
                size_t lim = std::min(kMax, n - i), l = 0;
                while (l < lim && raw[(size_t)cand + l] == raw[i + l]) l++;
                if (l > bestLen) {
                    bestLen = l;
                    bestDist = d;
                    if (l >= lim) break;
                }
                int32_t nx = prev[(size_t)cand & (kWin - 1)];
                if (nx >= cand) break;
                cand = nx;
            }
        }
        if (bestLen >= 3) {
            int li = 0;
            while (li < 28 && lenBase[li + 1] <= bestLen) li++;
            // fixed Huffman: symbols 257-279 are 7-bit codes 0000001..0010111, symbols 280-285
            // are 8-bit codes 11000000..11000101 (RFC 1951 3.2.6); the canonical code value is
            // not simply symbol-256, so map both ranges explicitly.
            if (li < 23) putCode(257u + (uint32_t)li, 7);
            else putCode(192u + (uint32_t)(li - 23), 8);
            if (lenExtra[li]) putBits((uint32_t)(bestLen - lenBase[li]), lenExtra[li]);
            int di = 0;
            while (di < 29 && distBase[di + 1] <= bestDist) di++;
            putCode((uint32_t)di, 5);
            if (distExtra[di]) putBits((uint32_t)(bestDist - distBase[di]), distExtra[di]);
            for (size_t k = 0; k < bestLen; k++) insert(i + k);
            i += bestLen;
        } else {
            uint32_t v = raw[i];
            if (v < 144) putCode(0x30u + v, 8);
            else putCode(0x190u + (v - 144), 9);
            insert(i);
            i++;
        }
    }
    putCode(0, 7);  // end of block
    if (bitCnt) body.push_back((uint8_t)bitBuf);
    return body;
}

static bool pngWriteRaw(const std::string& path, const std::vector<uint8_t>& rgb, uint32_t w, uint32_t h,
                        int compress, const std::string& info) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    std::fwrite(sig, 1, 8, f);
    uint8_t ihdr[13];  // width, height, 8-bit, RGB, no compression/filter/interlace
    ihdr[0] = (uint8_t)(w >> 24);
    ihdr[1] = (uint8_t)(w >> 16);
    ihdr[2] = (uint8_t)(w >> 8);
    ihdr[3] = (uint8_t)w;
    ihdr[4] = (uint8_t)(h >> 24);
    ihdr[5] = (uint8_t)(h >> 16);
    ihdr[6] = (uint8_t)(h >> 8);
    ihdr[7] = (uint8_t)h;
    ihdr[8] = 8;
    ihdr[9] = 2;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    pngChunk(f, "IHDR", ihdr, 13);
    if (!info.empty()) pngChunk(f, "tEXt", (const uint8_t*)info.data(), (uint32_t)info.size());
    size_t stride = (size_t)w * 3 + 1, rows = h;
    std::vector<uint8_t> raw((size_t)h * stride);
    for (uint32_t y = 0; y < h; y++) {
        uint8_t* r = &raw[(size_t)y * stride];
        r[0] = 0;  // filter: none
        std::memcpy(r + 1, rgb.data() + (size_t)y * w * 3, (size_t)w * 3);
    }
    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    if (compress) {
        // deflateFixed returns a complete raw deflate stream (BFINAL block + EOB), not payload.
        std::vector<uint8_t> body = deflateFixed(raw);
        z.insert(z.end(), body.begin(), body.end());
    } else {
        size_t pos = 0;
        while (pos < raw.size()) {
            size_t k = std::min<size_t>(65535, raw.size() - pos);
            z.push_back(pos + k == raw.size() ? 1 : 0);
            z.push_back((uint8_t)(k & 0xFF));
            z.push_back((uint8_t)(k >> 8));
            z.push_back((uint8_t)(~k & 0xFF));
            z.push_back((uint8_t)((~k >> 8) & 0xFF));
            z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + k);
            pos += k;
        }
    }
    uint32_t a1 = 1, a2 = 0;  // adler32
    for (uint8_t b : raw) {
        a1 = (a1 + b) % 65521;
        a2 = (a2 + a1) % 65521;
    }
    uint32_t ad = (a2 << 16) | a1;
    z.push_back((uint8_t)(ad >> 24));
    z.push_back((uint8_t)(ad >> 16));
    z.push_back((uint8_t)(ad >> 8));
    z.push_back((uint8_t)ad);
    pngChunk(f, "IDAT", z.data(), (uint32_t)z.size());
    pngChunk(f, "IEND", nullptr, 0);
    bool ok = !std::ferror(f);
    std::fclose(f);
    (void)rows;
    return ok;
}

// CHW floats in [-1,1] (the decoder's output space) to an 8-bit RGB file.
static bool writeImageFile(const std::string& path, const std::vector<float>& img, uint32_t w, uint32_t h,
                           int compress, const std::string& info) {
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            for (uint32_t c = 0; c < 3; c++) {
                float v = img[((uint64_t)c * h + y) * w + x] * 0.5f + 0.5f;
                v = std::max(0.0f, std::min(1.0f, v));
                rgb[((size_t)y * w + x) * 3 + c] = (uint8_t)std::lround(v * 255.0f);
            }
        }
    }
    bool png = path.size() >= 4 && (path.compare(path.size() - 4, 4, ".png") == 0 ||
                                   path.compare(path.size() - 4, 4, ".PNG") == 0);
    if (png) return pngWriteRaw(path, rgb, w, h, compress, info);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    std::fwrite(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
    return true;
}

// ================================================================ latent IO

static bool readLatent(const std::string& path, std::vector<float>& lat, uint32_t& h, uint32_t& w, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    char magic[4] = {0};
    uint32_t d[3] = {0, 0, 0};
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::fread(d, 4, 3, f) == 3 && !std::memcmp(magic, "OFCL", 4) &&
              d[0] == 4 && d[1] > 0 && d[2] > 0;
    if (ok) {
        lat.resize((size_t)d[0] * d[1] * d[2]);
        ok = std::fread(lat.data(), 4, lat.size(), f) == lat.size();
    }
    std::fclose(f);
    if (!ok) {
        err = "bad latent file " + path + " (expect OFCL magic + dims {4,H,W})";
        return false;
    }
    h = d[1];
    w = d[2];
    return true;
}

static bool writeLatent(const std::string& path, const std::vector<float>& lat, uint32_t h, uint32_t w) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const char magic[4] = {'O', 'F', 'C', 'L'};
    std::fwrite(magic, 1, 4, f);
    // dims are {channels, H, W}: writing the same value for both turned every portrait latent into
    // a square on reload, and the decoder then produced a squashed image
    uint32_t d[3] = {4, h, w};
    std::fwrite(d, 4, 3, f);
    std::fwrite(lat.data(), 4, lat.size(), f);
    std::fclose(f);
    return true;
}

// ================================================================ sampler

struct Euler {
    std::vector<float> timesteps;
    std::vector<float> sigmas;
    float initNoiseSigma = 1.0f;
};

// diffusers EulerDiscrete::set_timesteps: "leading" (SD1.5 style, steps_offset=1) and "trailing"
// (what the SDXL repos ship). The two differ in which timesteps get sampled AND in the init scale.
// zsnr = rescale_betas_zero_snr (zero terminal SNR, the recommended schedule for v-pred SDXL):
// sqrt(alpha_bar) is shifted so the terminal entry is 0 and rescaled so the first is unchanged;
// the terminal alpha_bar is then clamped to 2^-24 (sigma_max ~4096). The t fed to the model is
// unchanged - only the sigma levels move.
static float sigmaAt(const std::vector<float>& sigmaOf, float t, int numTrain) {
    if (t <= 0) return sigmaOf[0];
    if (t >= numTrain - 1) return sigmaOf[numTrain - 1];
    int i0 = (int)t;
    float fr = t - (float)i0;
    return sigmaOf[i0] * (1.0f - fr) + sigmaOf[i0 + 1] * fr;
}

static Euler buildEuler(int steps, const char* spacing, int numTrain, int offset, bool zsnr) {
    std::vector<float> betas(numTrain);
    for (int i = 0; i < numTrain; i++) {
        float b0 = 0.00085f, b1 = 0.012f;
        float v = std::sqrt(b0) + (std::sqrt(b1) - std::sqrt(b0)) * (float)i / (float)(numTrain - 1);
        betas[i] = v * v;
    }
    std::vector<double> ab(numTrain);
    double run = 1.0;
    for (int i = 0; i < numTrain; i++) {
        run *= (1.0 - betas[i]);
        ab[i] = run;
    }
    if (zsnr) {
        double a0 = std::sqrt(ab[0]), aT = std::sqrt(ab[numTrain - 1]);
        for (int i = 0; i < numTrain; i++) {
            double s = (std::sqrt(ab[i]) - aT) * (a0 / (a0 - aT));
            ab[i] = s * s;
        }
        ab[numTrain - 1] = std::pow(2.0, -24.0);
    }
    std::vector<float> sigmaOf(numTrain);
    for (int i = 0; i < numTrain; i++) sigmaOf[i] = (float)std::sqrt((1.0 - ab[i]) / ab[i]);
    Euler e;
    bool trailing = std::strcmp(spacing, "trailing") == 0;
    e.timesteps.resize(steps);
    if (trailing) {
        float stepRatio = (float)numTrain / (float)steps;
        for (int i = 0; i < steps; i++)
            e.timesteps[i] = std::floor((float)numTrain - (float)i * stepRatio + 0.5f) - 1.0f;
    } else {
        int stepRatio = numTrain / steps;
        for (int i = 0; i < steps; i++)
            e.timesteps[steps - 1 - i] = (float)std::lround((float)i * stepRatio) + (float)offset;
    }
    e.sigmas.resize(steps + 1);
    for (int i = 0; i < steps; i++) e.sigmas[i] = sigmaAt(sigmaOf, e.timesteps[i], numTrain);
    e.sigmas[steps] = 0.0f;
    float mx = 0.0f;
    for (float s : e.sigmas) mx = std::max(mx, s);
    e.initNoiseSigma = trailing ? mx : std::sqrt(mx * mx + 1.0f);
    return e;
}

// sigma levels for a partial (img2img / inpainting) run: start from the level that matches the
// requested strength and keep `steps` evenly spaced sigmas down to zero
static Euler buildPartial(const Euler& full, int steps, float denoise, bool zsnr) {
    Euler e;
    e.initNoiseSigma = full.initNoiseSigma;
    float smax = full.sigmas[0];
    std::vector<float> inv;
    for (float s : full.sigmas) inv.push_back(s);
    std::vector<float> target(inv.rbegin(), inv.rend());  // ascending, 0 .. smax
    float s0 = smax * denoise;
    int n = std::max(1, steps);
    e.sigmas.assign(n + 1, 0.0f);
    for (int i = 0; i < n; i++) {
        float want = s0 * (1.0f - (float)i / (float)n);
        size_t k = 0;
        while (k + 1 < target.size() && target[k + 1] <= want) k++;
        float frac = 0.0f;
        if (k + 1 < target.size() && target[k + 1] > target[k]) frac = (want - target[k]) / (target[k + 1] - target[k]);
        e.sigmas[i] = target[k] + (target[std::min(k + 1, target.size() - 1)] - target[k]) * frac;
    }
    e.sigmas[n] = 0.0f;
    // t fed to the model must follow the same sigma, so map back through the schedule
    e.timesteps.assign(n, 0.0f);
    for (int i = 0; i < n; i++) {
        float s = e.sigmas[i];
        size_t best = 0;
        float bd = 1e30f;
        for (size_t k = 0; k < full.sigmas.size(); k++) {
            float d = std::fabs(full.sigmas[k] - s);
            if (d < bd) {
                bd = d;
                best = k;
            }
        }
        e.timesteps[i] = full.timesteps[std::min(best, full.timesteps.size() - 1)];
        if (i > 0 && e.timesteps[i] > e.timesteps[i - 1]) e.timesteps[i] = e.timesteps[i - 1];
    }
    if (zsnr) e.timesteps[0] = std::min(e.timesteps[0], full.timesteps[0]);
    return e;
}

// sigma_up for ancestral sampling (diffusers: eta * sqrt(sigma_down^2 * (sigma^2 - sigma_down^2) / sigma^2))
static float ancestralSigma(float sDown, float sigma, float eta) {
    if (sigma <= 0.0f) return 0.0f;
    float inside = sDown * sDown * (sigma * sigma - sDown * sDown) / (sigma * sigma);
    return eta * std::sqrt(std::max(0.0f, inside));
}

// ================================================================ runtime state

namespace ofc {
namespace sdxl {

namespace {

struct Tokenizers {
    bool ready = false;
    std::string dir1, dir2;
    TokSet t1, t2;
};

struct State {
    Tokenizers tok;
    std::string spvDir;
    bool unetResident = false;
    double unetMs = 0;
    double vaeMs = 0;
};

State gS;

}  // namespace

static void report(const ProgressFn& fn, Progress::Kind k, int step, int steps, double frac, double ms,
                   const std::string& text) {
    if (!fn) return;
    Progress p;
    p.kind = k;
    p.step = step;
    p.steps = steps;
    p.progress = std::max(0.0, std::min(1.0, frac));
    p.ms = ms;
    p.text = text;
    fn(p);
}

void unloadModel(EvalModel& m) {
    // A VAE-only decode leaves the engine alive without m.loaded, so "not loaded" is not the
    // same as "no device memory". Tear down whenever the engine still owns an instance.
    if (!m.loaded && !G.eng.alive()) {
        delete (State*)m.impl;
        m.impl = nullptr;
        return;
    }
    if (G.eng.alive()) gShutdown();
    clearWeights();
    gS.unetResident = false;
    gS.tok.ready = false;
    m.loaded = false;
    m.unetPath.clear();
    m.vaePath.clear();
    delete (State*)m.impl;
    m.impl = nullptr;
}

bool ensureDevice(const std::string& spvDir, std::string& err, int device = 0) {
    try {
        if (G.eng.alive()) return true;
        G.eng.setDeviceIndex(device < 0 ? 0 : device);
        G.paramBytes = 64ull << 20;
        G.allocGradArena = false;
        G.stepBytes = 64ull << 20;
        G.scratchBytes = 64ull << 20;
        gInit(spvDir);
        gDeviceName = G.eng.deviceName();
        return true;
    } catch (const std::exception& e) {
        err = std::string("device init failed: ") + e.what();
        return false;
    }
}

// ================================================================ model load

// 6400 MB used to be unconditional, which cannot fit an 8 GB card once the desktop holds its
// share. Size the arena from the file instead: a Q8_0 GGUF lands near its byte size, a
// safetensors UNet is fp16 on disk and fp32 in the arena, so it needs about twice as much.
static uint64_t unetParamBytes(const GenParams& p) {
    uint64_t mb = 0;
    std::error_code ec;
    uint64_t fileBytes = std::filesystem::file_size(p.models.unet, ec);
    if (ec) fileBytes = 0;
    bool gguf = p.models.unet.size() > 5 && p.models.unet.compare(p.models.unet.size() - 5, 5, ".gguf") == 0;
    // Q8_0 keeps its byte size; fp16 safetensors expands to fp32, and the split Q8 payload /
    // fp32 scale pairs add a little on top. The margin covers alignment and the small side
    // tables; a 288 MB margin once came up 0.4 MB short, so leave a little more.
    mb = gguf ? ((fileBytes + (320ull << 20)) >> 20) : ((fileBytes / 1048576) * 2 + 640);
    if (gguf) {
        // A full-checkpoint GGUF also carries CLIP and the VAE. Sizing the arena from the file
        // would over-allocate by ~1.5 GB, which on an 8 GB card makes the AMD driver kill the
        // process outright. Sum the UNet tensors instead.
        GgufFile probe;
        std::string perr;
        bool full = probe.open(p.models.unet, &perr) &&
                    probe.find("model.diffusion_model.input_blocks.0.0.weight") != nullptr;
        if (full && probe.open(p.models.unet, &perr, "model.diffusion_model.")) {
            uint64_t need = 0;
            for (auto& t : probe.tensors()) {
                uint64_t ne = t.nElems();
                need += t.type == GgmlType::Q8_0 ? ne + (ne / 32) * 4 : ne * 4;
            }
            mb = (need + (192ull << 20)) >> 20;
        }
    }
    if (!gguf) mb = std::max<uint64_t>(mb, 512);
    mb = std::max<uint64_t>(mb, 512);
    mb = std::min<uint64_t>(mb, 6144);
    // --param-mb is a floor, like --step-mb: asking for less than the weights need can only fail
    // halfway through the upload, so take the larger value and say so.
    if (p.paramMb && p.paramMb < mb) {
        std::printf("note: --param-mb %u MB is below the %llu MB this file needs; using %llu MB\n", p.paramMb,
                    (unsigned long long)mb, (unsigned long long)mb);
    }
    return std::max<uint64_t>(p.paramMb, mb) << 20;
}

// fp32 footprint of CLIP-L + CLIP-G, rounded up. An fp16 safetensors file is half of it; a
// single-file checkpoint shares one file between both encoders, so fall back to the old 3.4 GB.
static uint64_t teParamBytes(const GenParams& p) {
    uint64_t mb = 0;
    if (p.teParamMb) return (uint64_t)p.teParamMb << 20;
    std::error_code ec;
    uint64_t a = std::filesystem::file_size(p.models.te1, ec);
    if (ec) a = 0;
    uint64_t b = p.models.te2 == p.models.te1 ? 0 : std::filesystem::file_size(p.models.te2, ec);
    if (ec) b = 0;
    mb = (a + b) * 2 / 1048576 + 512;
    return (mb ? std::min<uint64_t>(mb, 6144) : 3400) << 20;
}

bool loadModel(EvalModel& m, const GenParams& in, const ProgressFn& onProgress, std::string& err) {    GenParams p = in;
    if (p.spvDir.empty()) p.spvDir = "shaders";
    ModelSet set;
    if (!ofc::sdxl::resolveModels(p, set, err)) return false;
    if (!ensureDevice(p.spvDir, err, p.device)) return false;
    gLora.clear();
    if (!p.lora.empty()) loadLoraFile(p.lora, p.loraScale);
    gS.spvDir = p.spvDir;
    gS.tok.ready = false;
    m.warnings = set.warn;
    for (auto& w : set.warn) {
        std::printf("note: %s\n", w.c_str());
        report(onProgress, Progress::Kind::Warning, 0, 0, 0.0, 0.0, w);
    }
    try {
        if (gS.unetResident && gUNet.path == p.models.unet) {
            m.loaded = true;
            m.deviceName = gDeviceName;
            m.modelName = set.unetName;
            m.unetPath = p.models.unet;
            m.vaePath = p.models.vae;
            report(onProgress, Progress::Kind::Load, 0, 0, 0.02, 0.0, "UNet already resident");
            return true;
        }
        report(onProgress, Progress::Kind::Load, 0, 0, 0.02, 0.0, "loading UNet");
        clearWeights();
        G.paramBytes = unetParamBytes(p);
        G.eng.initArenas(256ull << 20, 256ull << 20);
        G.paramArena.init(G.eng.device(), G.eng.physDevice(), G.paramBytes, true, false);
        std::printf("unet arena: %.0f MB\n", G.paramBytes / 1048576.0);
        auto t0 = std::chrono::steady_clock::now();
        loadUnet(p);
        gS.unetMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        gS.unetResident = true;
        m.loaded = true;
        m.unetPath = p.models.unet;
        m.vaePath = p.models.vae;
        m.deviceName = gDeviceName;
        m.modelName = set.unetName;
        report(onProgress, Progress::Kind::Load, 0, 0, 0.08, gS.unetMs, "UNet ready");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        unloadModel(m);
        return false;
    }
}

// ================================================================ text / init images

static void loadTokenizers(Tokenizers& t, const GenParams& p) {
    if (t.ready && t.dir1 == p.models.tok1 && t.dir2 == p.models.tok2) return;
    t.t1 = TokSet();
    t.t2 = TokSet();
    buildByteMap(t.t1);
    buildByteMap(t.t2);
    loadVocab(t.t1, p.models.tok1 + "/vocab.json");
    loadMerges(t.t1, p.models.tok1 + "/merges.txt");
    loadVocab(t.t2, p.models.tok2 + "/vocab.json");
    loadMerges(t.t2, p.models.tok2 + "/merges.txt");
    if (t.t1.vocab.empty() || t.t1.merges.empty()) fail("tokenizer 1 is empty: " + p.models.tok1);
    if (t.t2.vocab.empty() || t.t2.merges.empty()) fail("tokenizer 2 is empty: " + p.models.tok2);
    std::printf("tokenizer: vocab %zu/%zu merges %zu/%zu\n", t.t1.vocab.size(), t.t2.vocab.size(),
                t.t1.merges.size(), t.t2.merges.size());
    t.dir1 = p.models.tok1;
    t.dir2 = p.models.tok2;
    t.ready = true;
}

// -------------------------------- PPM / minimal PNG reader (img2img seed)

static bool readPpm(const std::string& path, std::vector<uint8_t>& rgb, uint32_t& w, uint32_t& h) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[3] = {0, 0, 0};
    if (std::fscanf(f, "%2s", magic) != 1 || std::strcmp(magic, "P6") != 0) {
        std::fclose(f);
        return false;
    }
    int maxv = 0;
    if (std::fscanf(f, "%u %u %d", &w, &h, &maxv) != 3 || w == 0 || h == 0) {
        std::fclose(f);
        return false;
    }
    std::fgetc(f);  // single whitespace after the header
    rgb.resize((size_t)w * h * 3);
    bool ok = std::fread(rgb.data(), 1, rgb.size(), f) == rgb.size();
    std::fclose(f);
    return ok;
}

static uint32_t be32(const uint8_t* p) {
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | (uint32_t)p[3];
}

// Inflate for fixed/dynamic-Huffman or stored deflate streams (used to read img2img sources).
static bool inflateRaw(const uint8_t* src, size_t n, std::vector<uint8_t>& out) {
    struct Bits {
        const uint8_t* p;
        size_t n, pos = 0;
        uint32_t buf = 0;
        int cnt = 0;
        bool get(int k, uint32_t& v) {
            while (cnt < k) {
                if (pos >= n) return false;
                buf |= (uint32_t)p[pos++] << cnt;
                cnt += 8;
            }
            v = buf & (((uint32_t)1 << k) - 1);
            buf >>= k;
            cnt -= k;
            return true;
        }
    } bs{src, n};
    struct Huff {
        std::vector<uint16_t> count, symbol;
        bool build(const std::vector<uint8_t>& len) {
            count.assign(16, 0);
            symbol.assign(len.size(), 0);
            for (uint8_t l : len) count[l]++;
            if (count[0] == len.size()) return true;
            std::vector<uint16_t> off(16, 0);
            for (int i = 1; i < 16; i++) off[i] = off[i - 1] + count[i - 1];
            for (size_t i = 0; i < len.size(); i++)
                if (len[i]) symbol[off[len[i]]++] = (uint16_t)i;
            return true;
        }
        bool decode(Bits& b, int& sym) const {
            int code = 0, first = 0, index = 0;
            for (int len = 1; len < 16; len++) {
                uint32_t bit;
                if (!b.get(1, bit)) return false;
                code |= (int)bit;
                int c = count[len];
                if (code - c < first) {
                    sym = symbol[index + (code - first)];
                    return true;
                }
                index += c;
                first = (first + c) << 1;
                code <<= 1;
            }
            return false;
        }
    };
    static const uint16_t lbase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                       31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static const uint8_t lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                     2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static const uint16_t dbase[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,
                                       33,  49,  65,  97,  129, 193,  257,  385,  513,  769,
                                       1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static const uint8_t dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                                     6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    int last = 0;
    while (!last) {
        uint32_t v;
        if (!bs.get(1, v)) return false;
        last = (int)v;
        if (!bs.get(2, v)) return false;
        int type = (int)v;
        if (type == 0) {
            bs.cnt = 0;
            bs.buf = 0;
            if (bs.pos + 4 > bs.n) return false;
            uint32_t len = src[bs.pos] | (src[bs.pos + 1] << 8);
            bs.pos += 4;
            if (bs.pos + len > bs.n) return false;
            out.insert(out.end(), src + bs.pos, src + bs.pos + len);
            bs.pos += len;
            continue;
        }
        Huff lit, dist;
        if (type == 1) {
            std::vector<uint8_t> l(288);
            for (int i = 0; i < 144; i++) l[i] = 8;
            for (int i = 144; i < 256; i++) l[i] = 9;
            for (int i = 256; i < 280; i++) l[i] = 7;
            for (int i = 280; i < 288; i++) l[i] = 8;
            lit.build(l);
            std::vector<uint8_t> d(30, 5);
            dist.build(d);
        } else if (type == 2) {
            uint32_t hlit, hdist, hclen;
            if (!bs.get(5, hlit) || !bs.get(5, hdist) || !bs.get(4, hclen)) return false;
            static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            std::vector<uint8_t> cl(19, 0);
            for (uint32_t i = 0; i < hclen + 4; i++) {
                uint32_t b;
                if (!bs.get(3, b)) return false;
                cl[order[i]] = (uint8_t)b;
            }
            Huff ch;
            ch.build(cl);
            std::vector<uint8_t> lens;
            while (lens.size() < hlit + 257 + hdist + 1) {
                int sym;
                if (!ch.decode(bs, sym)) return false;
                if (sym < 16) {
                    lens.push_back((uint8_t)sym);
                } else {
                    uint32_t rep = 0, val = 0;
                    if (sym == 16) {
                        if (lens.empty()) return false;
                        val = lens.back();
                        if (!bs.get(2, rep)) return false;
                        rep += 3;
                    } else if (sym == 17) {
                        if (!bs.get(3, rep)) return false;
                        rep += 3;
                    } else {
                        if (!bs.get(7, rep)) return false;
                        rep += 11;
                    }
                    for (uint32_t i = 0; i < rep; i++) lens.push_back((uint8_t)val);
                }
            }
            std::vector<uint8_t> l(lens.begin(), lens.begin() + hlit + 257);
            std::vector<uint8_t> d(lens.begin() + hlit + 257, lens.begin() + hlit + 257 + hdist + 1);
            lit.build(l);
            dist.build(d);
        } else {
            return false;
        }
        while (true) {
            int sym;
            if (!lit.decode(bs, sym)) return false;
            if (sym < 256) {
                out.push_back((uint8_t)sym);
            } else if (sym == 256) {
                break;
            } else {
                int li = sym - 257;
                if (li < 0 || li > 28) return false;
                uint32_t ex = 0;
                if (lext[li] && !bs.get(lext[li], ex)) return false;
                size_t len = lbase[li] + ex;
                int ds;
                if (!dist.decode(bs, ds) || ds > 29) return false;
                uint32_t dx = 0;
                if (dext[ds] && !bs.get(dext[ds], dx)) return false;
                size_t d = dbase[ds] + dx;
                if (d > out.size()) return false;
                for (size_t i = 0; i < len; i++) out.push_back(out[out.size() - d]);
            }
        }
    }
    return true;
}

static bool readPng(const std::string& path, std::vector<uint8_t>& rgb, uint32_t& w, uint32_t& h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (file.size() < 8 || be32(file.data()) != 0x89504E47) return false;
    size_t pos = 8;
    uint32_t width = 0, height = 0;
    int bitDepth = 0, colorType = 0;
    std::vector<uint8_t> idat;
    while (pos + 8 <= file.size()) {
        uint32_t len = be32(&file[pos]);
        std::string type((const char*)&file[pos + 4], 4);
        if (pos + 12 + len > file.size()) return false;
        const uint8_t* d = &file[pos + 8];
        if (type == "IHDR") {
            width = be32(d);
            height = be32(d + 4);
            bitDepth = d[8];
            colorType = d[9];
            if (d[12] != 0) return false;  // interlaced PNGs are not supported as init images
        } else if (type == "IDAT") {
            idat.insert(idat.end(), d, d + len);
        } else if (type == "IEND") {
            break;
        }
        pos += 12 + len;
    }
    if (!width || !height || bitDepth != 8) return false;
    int ch = colorType == 2 ? 3 : colorType == 6 ? 4 : colorType == 0 ? 1 : 0;
    if (!ch) return false;
    std::vector<uint8_t> raw;
    if (!inflateRaw(idat.data(), idat.size(), raw)) return false;
    size_t stride = (size_t)width * ch;
    if (raw.size() < (stride + 1) * height) return false;
    std::vector<uint8_t> px(stride * height);
    auto paeth = [](int a, int b, int c) {
        int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
        return (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
    };
    for (uint32_t y = 0; y < height; y++) {
        int filt = raw[y * (stride + 1)];
        const uint8_t* srcRow = &raw[y * (stride + 1) + 1];
        uint8_t* outRow = &px[y * stride];
        const uint8_t* prevRow = y ? &px[(y - 1) * stride] : nullptr;
        for (size_t x = 0; x < stride; x++) {
            int a = x >= (size_t)ch ? outRow[x - ch] : 0;
            int b = prevRow ? prevRow[x] : 0;
            int c = (prevRow && x >= (size_t)ch) ? prevRow[x - ch] : 0;
            int v = srcRow[x];
            switch (filt) {
                case 1:
                    v += a;
                    break;
                case 2:
                    v += b;
                    break;
                case 3:
                    v += (a + b) / 2;
                    break;
                case 4:
                    v += paeth(a, b, c);
                    break;
                default:
                    break;
            }
            outRow[x] = (uint8_t)v;
        }
    }
    rgb.resize((size_t)width * height * 3);
    for (size_t i = 0; i < (size_t)width * height; i++) {
        if (ch >= 3) {
            rgb[i * 3] = px[i * ch];
            rgb[i * 3 + 1] = px[i * ch + 1];
            rgb[i * 3 + 2] = px[i * ch + 2];
        } else {
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = px[i * ch];
        }
    }
    w = width;
    h = height;
    return true;
}

// Nearest-neighbour resample to the target size, then CHW floats in [-1,1].
static std::vector<float> resampleChw(const std::vector<uint8_t>& rgb, uint32_t sw, uint32_t sh, uint32_t dw,
                                      uint32_t dh) {
    std::vector<float> out((size_t)3 * dw * dh);
    for (uint32_t y = 0; y < dh; y++) {
        uint32_t sy = (uint32_t)((uint64_t)y * sh / dh);
        for (uint32_t x = 0; x < dw; x++) {
            uint32_t sx = (uint32_t)((uint64_t)x * sw / dw);
            const uint8_t* p = &rgb[((size_t)sy * sw + sx) * 3];
            for (int c = 0; c < 3; c++) out[((size_t)c * dh + y) * dw + x] = p[c] / 255.0f * 2.0f - 1.0f;
        }
    }
    return out;
}

// ---------------------------------------------------------------- inpaint mask

// Separable box blur with clamped edges. Two passes approximate a triangular kernel, which is
// close enough to a gaussian for feathering an inpaint seam and costs O(H*W) per pass.
static void boxBlurMask(std::vector<float>& m, uint32_t H, uint32_t W, int r) {
    if (r <= 0 || H == 0 || W == 0) return;
    std::vector<float> t(m.size());
    std::vector<double> pre((size_t)std::max(H, W) + 1);
    for (uint32_t y = 0; y < H; y++) {
        const float* row = &m[(size_t)y * W];
        pre[0] = 0;
        for (uint32_t x = 0; x < W; x++) pre[x + 1] = pre[x] + row[x];
        for (uint32_t x = 0; x < W; x++) {
            int a = (int)x - r, b = (int)x + r + 1;
            int ca = std::max(0, a), cb = std::min((int)W, b);
            double sum = pre[cb] - pre[ca];
            int n = cb - ca;
            sum += (double)row[0] * (ca - a) + (double)row[W - 1] * (b - cb);
            n += (ca - a) + (b - cb);
            t[(size_t)y * W + x] = (float)(sum / (double)std::max(1, n));
        }
    }
    for (uint32_t x = 0; x < W; x++) {
        pre[0] = 0;
        for (uint32_t y = 0; y < H; y++) pre[y + 1] = pre[y] + t[(size_t)y * W + x];
        for (uint32_t y = 0; y < H; y++) {
            int a = (int)y - r, b = (int)y + r + 1;
            int ca = std::max(0, a), cb = std::min((int)H, b);
            double sum = pre[cb] - pre[ca];
            int n = cb - ca;
            sum += (double)t[x] * (ca - a) + (double)t[(size_t)(H - 1) * W + x] * (b - cb);
            n += (ca - a) + (b - cb);
            m[(size_t)y * W + x] = (float)(sum / (double)std::max(1, n));
        }
    }
    for (auto& v : m) v = std::min(1.0f, std::max(0.0f, v));
}

// White = repaint. The mask is box-averaged down to latent resolution (each latent cell covers
// 8x8 canvas pixels) so a half-covered cell lands halfway between keep and repaint instead of
// snapping, then feathered so the boundary does not show as a hard rectangle.
static std::vector<float> buildLatentMask(const std::string& path, uint32_t cw, uint32_t ch, uint32_t Lh,
                                          uint32_t Lw, uint32_t feather, bool invert) {
    std::vector<uint8_t> rgb;
    uint32_t mw = 0, mh = 0;
    if (!readPng(path, rgb, mw, mh) && !readPpm(path, rgb, mw, mh))
        fail("cannot read mask " + path + " (PNG or PPM)");
    if (!mw || !mh) fail("mask " + path + " is empty");
    std::vector<float> m((size_t)Lh * Lw, 1.0f);
    for (uint32_t ly = 0; ly < Lh; ly++) {
        for (uint32_t lx = 0; lx < Lw; lx++) {
            double acc = 0;
            for (uint32_t sy = 0; sy < 8; sy++) {
                uint32_t py = (uint32_t)std::min<uint64_t>((uint64_t)(ly * 8 + sy) * mh / ch, mh - 1);
                for (uint32_t sx = 0; sx < 8; sx++) {
                    uint32_t px = (uint32_t)std::min<uint64_t>((uint64_t)(lx * 8 + sx) * mw / cw, mw - 1);
                    const uint8_t* q = &rgb[((size_t)py * mw + px) * 3];
                    acc += (q[0] + q[1] + q[2]) / (3.0 * 255.0);
                }
            }
            float v = (float)(acc / 64.0);
            m[(size_t)ly * Lw + lx] = invert ? 1.0f - v : v;
        }
    }
    boxBlurMask(m, Lh, Lw, (int)feather);
    boxBlurMask(m, Lh, Lw, (int)feather);
    uint64_t painted = 0;
    for (float v : m)
        if (v > 0.5f) painted++;
    std::printf("inpaint mask %s: %ux%u -> %ux%u latent, feather %u, %.1f%% repainted\n", path.c_str(), mw, mh, Lw,
                Lh, feather, 100.0 * (double)painted / (double)m.size());
    return m;
}

// ================================================================ generate

// Peek at the UNet file before any text encoding: the TE stage needs to know whether a CLIP-G
// exists at all (SDXL) or not (SD1.5). loadUnet() re-detects from the loaded weights later.
static void detectArch(const std::string& unetPath) {
    if (unetPath.size() > 5 && unetPath.compare(unetPath.size() - 5, 5, ".gguf") == 0) return;
    Safetensors st;
    if (!st.open(unetPath)) return;
    bool sdxl = containsPrefix(st, "model.diffusion_model.label_emb.0.0.") || containsPrefix(st, "label_emb.0.0.");
    gArch = sdxl ? 2 : 1;
    std::printf("arch: %s\n", gArch == 1 ? "SD1.5" : "SDXL");
}

RunResult run(EvalModel& m, const GenParams& in, const ProgressFn& onProgress) {
    RunResult res;
    GenParams p = in;
    auto tStart = std::chrono::steady_clock::now();
    try {
        if (p.spvDir.empty()) p.spvDir = "shaders";
        // loadModel() resolves the model set on its own copy, so do it here as well: the text
        // encoder and VAE paths below are read from this p, not from the copy inside loadModel().
        {
            ModelSet set;
            std::string err;
            if (!ofc::sdxl::resolveModels(p, set, err)) throw std::runtime_error(err);
        }
        if (!m.loaded) {
            detectArch(p.models.unet);
            std::string err;
            if (!loadModel(m, p, onProgress, err)) throw std::runtime_error(err);
        }
        if (!gS.unetResident) throw std::runtime_error("model is not resident; call loadModel first");
        res.device = gDeviceName;
        res.modelName = m.modelName;
        res.width = p.width;
        res.height = p.heightOrWidth();
        res.seed = p.seed;

        // ---------------- text encoders
        uint32_t width = p.width, height = p.heightOrWidth();
        // CLIP attention peaks around 77x77x1280, so 192 MB is plenty; the sampler gets its own
        // bigger arenas later in this function.
        uint64_t teStepBytes = 192ull << 20;
        std::vector<float> embCond, embUncond, poolC, poolU;
        const uint32_t ctxDim = (gArch == 1) ? 768u : 2048u;  // SD1.5 cross-attn is CLIP-L only
        bool latinOnly = false;
        if (!p.embCondFile.empty()) {
            auto loadRaw = [](const std::string& path, size_t n, const char* what) {
                std::vector<float> v(n);
                FILE* f = std::fopen(path.c_str(), "rb");
                if (!f) fail(std::string("cannot open ") + what + " " + path);
                if (std::fread(v.data(), 4, n, f) != n) fail(std::string("short ") + path);
                std::fclose(f);
                return v;
            };
            embCond = loadRaw(p.embCondFile, 77ull * ctxDim, "emb-cond");
            embUncond = p.embUncondFile.empty() ? std::vector<float>(77ull * ctxDim, 0.0f)
                                                : loadRaw(p.embUncondFile, 77ull * ctxDim, "emb-uncond");
            poolC = p.embPoolFile.empty() ? std::vector<float>(1280, 0.0f) : loadRaw(p.embPoolFile, 1280, "emb-pool");
            poolU = p.embPoolUFile.empty() ? std::vector<float>(1280, 0.0f) : loadRaw(p.embPoolUFile, 1280, "emb-poolu");
            std::printf("emb from files (te phase skipped)\n");
        } else {
            // Only ~3.8 GB of device-local memory is available once the desktop takes its share.
            // CLIP-L and CLIP-G together are 2.65 GB of fp32 weights, which does not fit next to a
            // sampler-sized step arena, so the two encoders are loaded and run one at a time: each
            // pass binds its own weight arena, encodes, downloads, and the next pass rebuilds.
            loadTokenizers(gS.tok, p);
            std::vector<int> i1 = p.condTokens.empty() ? clipTokenize(gS.tok.t1, p.prompt) : p.condTokens;
            std::vector<int> i2 = p.condTokens.empty() ? clipTokenize(gS.tok.t2, p.prompt) : p.condTokens;
            if (p.condTokens.empty()) {
                size_t words = 0;
                for (size_t k = 1; k + 1 < i1.size() && i1[k] != 49407; k++) words++;
                latinOnly = words == 0;
            }
            // model_index.json force_zeros_for_empty_prompt=true: an empty negative prompt means zero
            // embeddings, not the empty string (its EOS context has rms 0.69 and wrecks CFG)
            bool needUncond = !p.negative.empty();
            std::vector<int> n1, n2;
            if (needUncond) {
                n1 = p.uncondTokens.empty() ? clipTokenize(gS.tok.t1, p.negative) : p.uncondTokens;
                n2 = p.uncondTokens.empty() ? clipTokenize(gS.tok.t2, p.negative) : p.uncondTokens;
            }
            embCond.assign(77ull * ctxDim, 0.0f);
            embUncond.assign(77ull * ctxDim, 0.0f);
            poolC.assign(1280, 0.0f);
            poolU.assign(1280, 0.0f);

            auto encodeOne = [&](int phase, const std::vector<int>& ids, std::vector<float>* pooled) {
                if (ids.size() != 77) fail("token arrays must hold 77 ids");
                bool wide = phase == 1;
                uint32_t D = wide ? 1280 : 768, heads = wide ? 32 : 12, hd = wide ? 20 : 12, layers = wide ? 4 : 3;
                std::vector<float> out(77ull * D);
                BufRef idx = G.scratchArena.alloc(77 * 4);
                BufRef outB = G.scratchArena.alloc(77ull * D * 4);
                BufRef outP = G.scratchArena.alloc(1280 * 4);
                G.eng.uploadNow(idx, ids.data(), 77 * 4);
                G.eng.beginStep();
                if (wide) clipEncode("te2.text_model.", "te2.text_projection", idx, ids, D, heads, hd, layers, outB, outP);
                else clipEncode("te1.text_model.", "", idx, ids, D, heads, hd, layers, outB, outP);
                G.eng.endStep();
                G.eng.waitStep();
                G.eng.downloadNow(outB, out.data(), out.size() * 4);
                if (pooled) G.eng.downloadNow(outP, pooled->data(), 1280 * 4);
                return out;
            };

            // CLIP-L first (small arena), then swap in CLIP-G (big arena) and encode again. Both
            // prompts go through each encoder so the 1.4 GB CLIP-G upload happens only once.
            report(onProgress, Progress::Kind::Encode, 0, 0, 0.10, 0.0, "encoding prompt");
            loadOneTextEncoder(p, 0);
            {
                std::vector<float> eL = encodeOne(0, i1, nullptr);
                std::vector<float> eLn = needUncond ? encodeOne(0, n1, nullptr) : std::vector<float>();
                for (uint32_t t = 0; t < 77; t++) {
                    std::memcpy(&embCond[(size_t)t * ctxDim], &eL[(size_t)t * 768], 768 * 4);
                    if (needUncond) std::memcpy(&embUncond[(size_t)t * ctxDim], &eLn[(size_t)t * 768], 768 * 4);
                }
            }
            if (gArch != 1) {  // SD1.5 has no CLIP-G: the cross-attn context is CLIP-L only
            report(onProgress, Progress::Kind::Encode, 0, 0, 0.14, 0.0, "encoding prompt (clip-g)");
            loadOneTextEncoder(p, 1);
            {
                std::vector<float> poolCv(1280, 0.0f), poolUv(1280, 0.0f);
                std::vector<float> eG = encodeOne(1, i2, &poolCv);
                std::vector<float> eGn = needUncond ? encodeOne(1, n2, &poolUv) : std::vector<float>();
                for (uint32_t t = 0; t < 77; t++) {
                    std::memcpy(&embCond[(size_t)t * ctxDim + 768], &eG[(size_t)t * 1280], 1280 * 4);
                    if (needUncond) std::memcpy(&embUncond[(size_t)t * ctxDim + 768], &eGn[(size_t)t * 1280], 1280 * 4);
                }
                poolC = poolCv;
                if (needUncond) poolU = poolUv;
            }
            }
        }
        res.latinOnly = latinOnly;
        double ne = 0, nc = 0, nu = 0;
        for (float v : embCond) ne += (double)v * v;
        for (float v : poolC) nc += (double)v * v;
        for (float v : poolU) nu += (double)v * v;
        std::printf("ctx cond rms %.5f  |pooled_c| %.4f |pooled_u| %.4f\n", std::sqrt(ne / embCond.size()), std::sqrt(nc),
                    std::sqrt(nu));
        if (latinOnly && !p.prompt.empty()) {
            report(onProgress, Progress::Kind::Warning, 0, 0, 0.12, 0.0,
                   "prompt produced no word tokens (the CLIP tokenizer only covers latin text and byte fallback "
                   "is limited); try English");
        }
        if (p.textcheck) {
            const char* names[4][2] = {{"sdvkgen_ctx_cond.f32", "cond"}, {"sdvkgen_ctx_uncond.f32", "uncond"},
                                       {"sdvkgen_pooled_cond.f32", "poolC"}, {"sdvkgen_pooled_uncond.f32", "poolU"}};
            const std::vector<float>* bufs[4] = {&embCond, &embUncond, &poolC, &poolU};
            for (int i = 0; i < 4; i++) {
                FILE* f = std::fopen(names[i][0], "wb");
                std::fwrite(bufs[i]->data(), 4, bufs[i]->size(), f);
                std::fclose(f);
            }
            std::printf("textcheck: wrote ctx/pooled dumps\n");
            res.ok = true;
            res.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
            return res;
        }

        // ---------------- img2img / inpaint: real VAE encode
        // The old path wrote pixels * scaling_factor straight into the latent (all four channels a
        // copy of one image), which is why anything below denoise 0.7 came out washed out and only
        // a near-full repaint looked right. The encoder weights were always in the file; the
        // loader filtered them out.
        std::vector<float> initLat;   // sampler-scale latent of the seed image
        std::vector<float> initMask;  // 1 = repaint, latent resolution; empty = no inpaint
        if (p.useInit && !p.initImage.empty()) {
            std::vector<uint8_t> rgb;
            uint32_t iw = 0, ih = 0;
            if (!readPng(p.initImage, rgb, iw, ih) && !readPpm(p.initImage, rgb, iw, ih))
                fail("cannot read init image " + p.initImage + " (PNG or PPM)");
            uint32_t cw = p.crop.empty() ? iw : p.crop.w;
            uint32_t chh = p.crop.empty() ? ih : p.crop.h;
            std::vector<uint8_t> crop((size_t)cw * chh * 3);
            for (uint32_t y = 0; y < chh; y++) {
                for (uint32_t x = 0; x < cw; x++) {
                    uint32_t sx = (p.crop.empty() ? x : p.crop.x + x);
                    uint32_t sy = (p.crop.empty() ? y : p.crop.y + y);
                    sx = std::min(sx, iw - 1);
                    sy = std::min(sy, ih - 1);
                    std::memcpy(&crop[((size_t)y * cw + x) * 3], &rgb[((size_t)sy * iw + sx) * 3], 3);
                }
            }
            std::vector<float> img = resampleChw(crop, cw, chh, width, height);
            report(onProgress, Progress::Kind::Load, 0, 1, 0.12, 0.0, "vae encode");
            prepareVaeArena(p);
            initLat = vaeEncodeChain(p, img, height, width);
            uint32_t Lh0 = height / 8, Lw0 = width / 8;
            if (!p.maskImage.empty())
                initMask = buildLatentMask(p.maskImage, width, height, Lh0, Lw0, p.maskFeather, p.maskInvert);
            if (p.roundtrip) {
                // encode -> decode with no sampling: the only check that proves the two halves
                // agree. Report the PSNR against the resampled source instead of eyeballing it.
                report(onProgress, Progress::Kind::Decode, 1, 1, 0.85, 0.0, "vae decode");
                std::vector<float> rt = vaeDecode(p, initLat, Lh0, Lw0);
                double se = 0;
                uint64_t n = std::min<uint64_t>(rt.size(), img.size());
                for (uint64_t k = 0; k < n; k++) {
                    double d = (double)rt[k] - (double)img[k];
                    se += d * d;
                }
                double mse = n ? se / (double)n : 1.0;
                double psnr = mse > 1e-12 ? 10.0 * std::log10(4.0 / mse) : 99.0;
                std::printf("roundtrip %ux%u: mse %.6f  psnr %.2f dB\n", width, height, mse, psnr);
                std::string info = p.infoOut ? ofc::sdxl::infoText(p, m.modelName, gDeviceName) : std::string();
                if (!writeImageFile(p.outPath, rt, width, height, p.pngCompress, info))
                    fail("cannot write " + p.outPath);
                res.outPath = p.outPath;
                res.width = width;
                res.height = height;
                res.image.width = width;
                res.image.height = height;
                res.image.rgb.resize((size_t)3 * width * height);
                for (uint64_t k = 0; k < res.image.rgb.size(); k++) res.image.rgb[k] = rt[k] * 0.5f + 0.5f;
                res.totalMs =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
                res.ok = true;
                report(onProgress, Progress::Kind::Done, 1, 1, 1.0, res.totalMs, "roundtrip done");
                return res;
            }
            std::printf("init image %s (%ux%u) -> vae encode %ux%u, denoise %.2f\n", p.initImage.c_str(), iw, ih,
                        Lw0, Lh0, p.denoise);
        }

        // ---------------- UNet
        // loadOneTextEncoder() rebound the weight arena for CLIP-G, so the UNet has to go back in
        // before sampling. stageUpload() renames that pass over the old contents, so this is safe.
        // The upload ring held the whole CLIP-G file; hand that RAM back before the 2.9 GB UNet
        // upload needs its own.
        G.eng.releaseUploadStaging();
        auto tU0 = std::chrono::steady_clock::now();
        clearWeights();
        G.paramBytes = unetParamBytes(p);
        G.paramArena.init(G.eng.device(), G.eng.physDevice(), G.paramBytes, true, false);
        loadUnet(p);
        uint32_t Lw = width / 8, Lh = height / 8;
        uint32_t stepMB = p.stepMb ? p.stepMb : (width >= 768 || height >= 768 ? 1500u : 900u);
        uint32_t scratchMB = p.scratchMb ? p.scratchMb : (width >= 768 || height >= 768 ? 900u : 420u);
        // Auto-size the activation pool from a stubbed allocation walk of the UNet (the counter
        // OFC_ARENA_PLAN prints). A hard-coded guess was 20 MB short at 768x1024+zsnr once and
        // killed a 30-step run; the walk costs a few ms and adapts to any canvas. The user's
        // --step-mb stays a floor; OFC_NO_AUTOSIZE=1 turns the walk off.
        uint32_t autoStepMB = 0;
        if (!std::getenv("OFC_NO_AUTOSIZE")) {
            gAutoPlan = true;
            gArenaPlan = true;
            planReset();
            {
                BufRef none;
                BufRef temb = buildTemb(none, none, none);
                (void)temb;
                unetForward(none, temb, none, Lh, Lw);
            }
            gAutoPlan = false;
            autoStepMB = (uint32_t)((gPlanPeak + (1ull << 20) - 1) >> 20) * 105 / 100 + 48;
            if (autoStepMB > stepMB) stepMB = autoStepMB;
        }
        // The driver pads each allocation and keeps the old one alive until its replacement is
        // bound, so a right-sized arena still loses to its predecessor. Give the activation pools
        // back before taking the new ones.
        G.eng.initArenas(32ull << 20, 32ull << 20);
        G.eng.initArenas((uint64_t)stepMB << 20, (uint64_t)scratchMB << 20);
        gResizeScratch((uint64_t)scratchMB << 20);
        gCol = G.scratchArena.alloc(kColFloats * 4);
        gSCap = 0;
        std::printf("arenas: step %u MB%s scratch %u MB\n", stepMB, autoStepMB ? " (auto-sized)" : "",
                    scratchMB);
        if (const char* env = std::getenv("OFC_ARENA_PLAN")) {
            if (env[0] && env[0] != '0') {
                gArenaPlan = true;
                planReset();
                std::printf("plan: arena planning on, nothing allocated\n");
            }
        }
        res.unetMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tU0).count();

        uint64_t latFloats = (uint64_t)4 * Lh * Lw;
        static std::unordered_map<std::string, BufRef> sCache;
        sCache.clear();
        auto scratchBuf = [&](const std::string& key, uint64_t bytes) {
            auto it = sCache.find(key);
            if (it != sCache.end()) return it->second;
            BufRef b = G.scratchArena.alloc((bytes + 255) & ~255ull);
            sCache[key] = b;
            return b;
        };
        BufRef ctxC = scratchBuf("__ctx_cond", (uint64_t)embCond.size() * 4);
        BufRef ctxU = scratchBuf("__ctx_uncond", (uint64_t)embUncond.size() * 4);
        BufRef poolCB = scratchBuf("__pool_cond", 1280 * 4);
        BufRef poolUB = scratchBuf("__pool_uncond", 1280 * 4);
        BufRef addTimeB = scratchBuf("__addtime", 1536 * 4);
        BufRef xLat = scratchBuf("__latent", latFloats * 4);
        BufRef embHost = scratchBuf("__temb", 320 * 4);
        G.eng.uploadNow(ctxC, embCond.data(), embCond.size() * 4);
        G.eng.uploadNow(ctxU, embUncond.data(), embUncond.size() * 4);
        G.eng.uploadNow(poolCB, poolC.data(), poolC.size() * 4);
        G.eng.uploadNow(poolUB, poolU.data(), poolU.size() * 4);

        std::vector<float> addTime(1536);
        float tv[6] = {(float)width, (float)height, 0.0f, 0.0f, (float)width, (float)height};
        for (int i = 0; i < 6; i++) {
            auto e = timeEmbHost(tv[i], 256);
            std::memcpy(&addTime[(size_t)i * 256], e.data(), 256 * 4);
        }
        G.eng.uploadNow(addTimeB, addTime.data(), addTime.size() * 4);

        // ---------------- probe mode
        if (p.probe) {
            std::vector<float> lat(latFloats);
            if (!p.probeX.empty()) {
                uint32_t ph = 0, pw = 0;
                std::string e;
                if (!readLatent(p.probeX, lat, ph, pw, e)) fail(e);
                if (ph != Lh || pw != Lw) fail("probe-x dim does not match the requested size");
            } else {
                std::mt19937 rng(p.seed);
                std::normal_distribution<float> nd(0.0f, 1.0f);
                for (auto& v : lat) v = nd(rng);
            }
            G.eng.uploadNow(xLat, lat.data(), lat.size() * 4);
            auto te = timeEmbHost(p.probeT, 320);
            G.eng.uploadNow(embHost, te.data(), te.size() * 4);
            G.eng.beginStep();
            BufRef temb = buildTemb(poolCB, addTimeB, embHost);
            gTrace = p.unetdbg;
            gDbSel = p.dbgStage.c_str();
            if (gTrace) tr("temb", temb, 1280);
            BufRef eps = unetForward(xLat, temb, ctxC, Lh, Lw);
            G.eng.endStep();
            G.eng.waitStep();
            flushTrace();
            gTrace = false;
            std::vector<float> ep(latFloats);
            G.eng.downloadNow(eps, ep.data(), ep.size() * 4);
            double rms = 0;
            uint64_t bad = 0;
            for (float v : ep) {
                if (!std::isfinite(v)) bad++;
                else rms += (double)v * v;
            }
            std::printf("probe: t=%.1f %ux%u eps rms %.5f bad %llu\n", p.probeT, Lw, Lh,
                        std::sqrt(rms / (double)ep.size()), (unsigned long long)bad);
            res.ok = true;
            res.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
            return res;
        }

        // ---------------- sampler schedule
        const char* sp = p.spacing.empty() ? "leading" : p.spacing.c_str();
        bool partial = p.useInit && p.denoise < 0.999f;
        int totalSteps = p.steps > 0 ? p.steps : 20;
        Euler full = buildEuler(totalSteps, sp, 1000, 1, p.zsnr);
        Euler sch = partial ? buildPartial(full, std::max(1, (int)std::lround(totalSteps * p.denoise)), p.denoise, p.zsnr)
                            : full;
        int steps = (int)sch.timesteps.size();
        res.steps = steps;
        std::printf("sampler %s: t0 %.1f  init_sigma %.5f  steps %d\n", ofc::sdxl::samplerName(p.sampler),
                    sch.timesteps[0], sch.initNoiseSigma, steps);

        std::mt19937 rng(p.seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> lat(latFloats);
        for (auto& v : lat) v = nd(rng) * sch.initNoiseSigma;
        bool seededFromImage = false;
        const uint64_t maskCells = (uint64_t)Lh * Lw;
        // The seed latent comes from the VAE encode done before the UNet was loaded; nothing is
        // re-read here, so img2img cannot silently fall back to the pixel path.
        if (!initLat.empty()) {
            uint64_t was = (uint64_t)4 * Lh * Lw;
            if (initLat.size() < was) fail("encoded latent does not match the canvas");
            for (uint64_t k = 0; k < latFloats; k++) lat[k] = initLat[k] + nd(rng) * sch.sigmas[0];
            seededFromImage = true;
            if (!initMask.empty()) std::printf("inpaint: latent mask blend active (%u px feather)\n", p.maskFeather);
        }
        G.eng.uploadNow(xLat, lat.data(), lat.size() * 4);

        auto runPass = [&](const BufRef& pool, const BufRef& ctx, float t, std::vector<float>& epsOut) {
            auto te = timeEmbHost(t, 320);
            G.eng.uploadNow(embHost, te.data(), te.size() * 4);
            G.eng.beginStep();
            BufRef temb = buildTemb(pool, addTimeB, embHost);
            BufRef eps = unetForward(xLat, temb, ctx, Lh, Lw);
            G.eng.endStep();
            G.eng.waitStep();
            G.eng.downloadNow(eps, epsOut.data(), epsOut.size() * 4);
        };

        bool needCfg = p.cfg > 1.0001f;
        std::vector<float> epsU(latFloats), epsC(latFloats), scaled(latFloats), resv(latFloats), x0v(latFloats),
            x0b(latFloats), epsG(latFloats), epsG2(latFloats), xTmp(latFloats), xNext(latFloats), absv(latFloats);
        double stepMsSum = 0;
        std::vector<std::vector<float>> lmsDeriv;
        std::vector<float> x0Prev(latFloats);
        bool havePrev = false;
        double tPrev = 0.0;

        auto x0FromEps = [&](const std::vector<float>& x, const std::vector<float>& e, float sig,
                             std::vector<float>& x0) {
            float s2 = sig * sig + 1.0f;
            float root = std::sqrt(s2);
            for (uint64_t k = 0; k < latFloats; k++)
                x0[k] = p.vpred ? e[k] * (-sig / root) + x[k] / s2 : x[k] - sig * e[k];
        };
        // Imagen/SD2.x dynamic thresholding plus the per-step inpaint blend. Both replace the x0
        // estimate, so they run on every model evaluation, not only on the final one of a step.
        auto finishX0 = [&](std::vector<float>& x0, float& thrOut) {
            float thr = 1.0f;
            if (p.dynthresh) {
                for (uint64_t k = 0; k < latFloats; k++) absv[k] = std::abs(x0[k]);
                size_t q = (size_t)((double)(latFloats - 1) * 0.995);
                std::nth_element(absv.begin(), absv.begin() + q, absv.end());
                thr = std::max(absv[q], 1.0f);
            }
            for (uint64_t k = 0; k < latFloats; k++) {
                float v = x0[k];
                if (p.dynthresh) v = std::min(std::max(v, -thr), thr) / thr;
                if (!initMask.empty()) {
                    float m = initMask[(size_t)(k % maskCells)];
                    v = m * v + (1.0f - m) * initLat[k];
                }
                x0[k] = v;
            }
            thrOut = thr;
        };

        for (int i = 0; i < steps; i++) {
            if (p.shouldCancel && p.shouldCancel()) fail("cancelled");
            float t = sch.timesteps[i];
            float sigma = sch.sigmas[i];
            float sigmaNext = sch.sigmas[i + 1];
            float dt = sigmaNext - sigma;
            double stepMs = 0;

            auto evalGuided = [&](const std::vector<float>& xIn, float tEv, float sigEv,
                                  std::vector<float>& outEps) {
                float invEv = 1.0f / std::sqrt(sigEv * sigEv + 1.0f);
                for (uint64_t k = 0; k < latFloats; k++) scaled[k] = xIn[k] * invEv;
                G.eng.uploadNow(xLat, scaled.data(), scaled.size() * 4);
                auto ta = std::chrono::steady_clock::now();
                runPass(poolCB, ctxC, tEv, epsC);
                if (needCfg) runPass(poolUB, ctxU, tEv, epsU);
                auto tb = std::chrono::steady_clock::now();
                double passMs = std::chrono::duration<double, std::milli>(tb - ta).count();
                stepMs += passMs;
                stepMsSum += passMs;

                uint64_t bad = 0;
                for (uint64_t k = 0; k < latFloats; k++)
                    if (!std::isfinite(epsC[k]) || (needCfg && !std::isfinite(epsU[k]))) bad++;
                if (bad) fail("non-finite model output at step " + std::to_string(i + 1) + " (" +
                              std::to_string(bad) + " of " + std::to_string(latFloats) + " values)");

                for (uint64_t k = 0; k < latFloats; k++)
                    resv[k] = needCfg ? epsU[k] + p.cfg * (epsC[k] - epsU[k]) : epsC[k];
                // CFG rescale ("mimic scale"): the guided output's std is clamped back toward the
                // uncond output's std before x0 is formed.
                if (p.rescale > 0.0f && needCfg) {
                    double m = 0, s2 = 0, u2 = 0;
                    for (uint64_t k = 0; k < latFloats; k++) m += resv[k];
                    m /= (double)latFloats;
                    for (uint64_t k = 0; k < latFloats; k++) {
                        double d = resv[k] - m;
                        s2 += d * d;
                    }
                    for (uint64_t k = 0; k < latFloats; k++) u2 += (double)epsU[k] * epsU[k];
                    double su = std::sqrt(u2 / (double)latFloats);
                    double sg = std::sqrt(s2 / (double)latFloats) + 1e-6;
                    for (uint64_t k = 0; k < latFloats; k++) {
                        float er = (float)((resv[k] - m) * (su / sg) + m);
                        resv[k] = (float)(p.rescale * er + (1.0 - p.rescale) * resv[k]);
                    }
                }
                std::copy(resv.begin(), resv.end(), outEps.begin());
            };

            evalGuided(lat, t, sigma, epsG);
            double se = 0;
            for (uint64_t k = 0; k < latFloats; k++) se += (double)epsG[k] * epsG[k];
            x0FromEps(lat, epsG, sigma, x0v);
            float thr = 1.0f;
            finishX0(x0v, thr);

            const float eta = p.eta > 0 ? p.eta : 1.0f;
            const bool last = sigmaNext <= 0.0f;
            const float tNext = last ? 0.0f : sch.timesteps[std::min(i + 1, steps - 1)];

            switch (p.sampler) {
                case SamplerKind::EulerAncestral: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    // diffusers EulerAncestral: step to sigma_down, then inject sigma_up noise.
                    float sigmaUp = ancestralSigma(sigmaNext, sigma, eta);
                    float sigmaDown = std::sqrt(std::max(0.0f, sigmaNext * sigmaNext - sigmaUp * sigmaUp));
                    float ratio = sigmaDown / sigma;
                    for (uint64_t k = 0; k < latFloats; k++)
                        xNext[k] = lat[k] + (lat[k] - x0v[k]) * (ratio - 1.0f) + nd(rng) * sigmaUp;
                    break;
                }
                case SamplerKind::Heun: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    for (uint64_t k = 0; k < latFloats; k++)
                        xTmp[k] = lat[k] + (lat[k] - x0v[k]) / sigma * dt;
                    evalGuided(xTmp, tNext, sigmaNext, epsG2);
                    x0FromEps(xTmp, epsG2, sigmaNext, x0b);
                    float thr2 = 1.0f;
                    finishX0(x0b, thr2);
                    for (uint64_t k = 0; k < latFloats; k++)
                        xNext[k] = lat[k] + 0.5f * ((lat[k] - x0v[k]) / sigma +
                                                    (xTmp[k] - x0b[k]) / sigmaNext) * dt;
                    break;
                }
                case SamplerKind::LMS: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    if (lmsDeriv.size() == 4) lmsDeriv.pop_back();
                    lmsDeriv.insert(lmsDeriv.begin(), std::vector<float>(latFloats));
                    for (uint64_t k = 0; k < latFloats; k++)
                        lmsDeriv.front()[k] = (lat[k] - x0v[k]) / sigma;
                    static const float coeff[4][4] = {
                        {1.0f, 0.0f, 0.0f, 0.0f},
                        {3.0f / 2.0f, -1.0f / 2.0f, 0.0f, 0.0f},
                        {23.0f / 12.0f, -16.0f / 12.0f, 5.0f / 12.0f, 0.0f},
                        {55.0f / 24.0f, -59.0f / 24.0f, 37.0f / 24.0f, -9.0f / 24.0f},
                    };
                    const float* cf = coeff[lmsDeriv.size() - 1];
                    for (uint64_t k = 0; k < latFloats; k++) {
                        float acc = 0.0f;
                        for (size_t j = 0; j < lmsDeriv.size(); j++) acc += cf[j] * lmsDeriv[j][k];
                        xNext[k] = lat[k] + dt * acc;
                    }
                    break;
                }
                case SamplerKind::DPM2: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    float sigmaMid = sigma + 0.5f * dt;
                    float tMid = 0.5f * (t + tNext);
                    for (uint64_t k = 0; k < latFloats; k++)
                        xTmp[k] = lat[k] + (lat[k] - x0v[k]) / sigma * (0.5f * dt);
                    evalGuided(xTmp, tMid, sigmaMid, epsG2);
                    x0FromEps(xTmp, epsG2, sigmaMid, x0b);
                    float thr2 = 1.0f;
                    finishX0(x0b, thr2);
                    for (uint64_t k = 0; k < latFloats; k++)
                        xNext[k] = lat[k] + (xTmp[k] - x0b[k]) / sigmaMid * dt;
                    break;
                }
                case SamplerKind::DPM2Ancestral: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    float sigmaUp = std::min(sigmaNext, ancestralSigma(sigmaNext, sigma, eta));
                    float sigmaDown = std::sqrt(std::max(0.0f, sigmaNext * sigmaNext - sigmaUp * sigmaUp));
                    float dDown = sigmaDown - sigma;
                    float sigmaMid = sigma + 0.5f * dDown;
                    float tMid = 0.5f * (t + tNext);
                    for (uint64_t k = 0; k < latFloats; k++)
                        xTmp[k] = lat[k] + (lat[k] - x0v[k]) / sigma * (0.5f * dDown);
                    evalGuided(xTmp, tMid, sigmaMid, epsG2);
                    x0FromEps(xTmp, epsG2, sigmaMid, x0b);
                    float thr2 = 1.0f;
                    finishX0(x0b, thr2);
                    for (uint64_t k = 0; k < latFloats; k++)
                        xNext[k] = lat[k] + (xTmp[k] - x0b[k]) / sigmaMid * dDown + nd(rng) * sigmaUp;
                    break;
                }
                case SamplerKind::DPMPP2M: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    double tNow = -std::log((double)sigma);
                    double tN = -std::log((double)sigmaNext);
                    double h = tN - tNow;
                    if (!havePrev) {
                        for (uint64_t k = 0; k < latFloats; k++)
                            xNext[k] = (sigmaNext / sigma) * lat[k] - std::expm1(-h) * x0v[k];
                    } else {
                        double r = (tNow - tPrev) / h;
                        float c0 = (float)(1.0 + 1.0 / (2.0 * r));
                        float c1 = (float)(1.0 / (2.0 * r));
                        for (uint64_t k = 0; k < latFloats; k++) {
                            float x0d = c0 * x0v[k] - c1 * x0Prev[k];
                            xNext[k] = (sigmaNext / sigma) * lat[k] - std::expm1(-h) * x0d;
                        }
                    }
                    x0Prev = x0v;
                    tPrev = tNow;
                    havePrev = true;
                    break;
                }
                case SamplerKind::DPMPP2SAncestral: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    float sigmaUp = std::min(sigmaNext, ancestralSigma(sigmaNext, sigma, eta));
                    float sigmaDown = std::sqrt(std::max(0.0f, sigmaNext * sigmaNext - sigmaUp * sigmaUp));
                    double tNow = -std::log((double)sigma);
                    double tN = -std::log((double)sigmaNext);
                    double h = tN - tNow;
                    if (!havePrev) {
                        for (uint64_t k = 0; k < latFloats; k++)
                            xNext[k] = (sigmaDown / sigma) * lat[k] - std::expm1(-h) * x0v[k];
                    } else {
                        double r = (tNow - tPrev) / h;
                        float c0 = (float)(1.0 + 1.0 / (2.0 * r));
                        float c1 = (float)(1.0 / (2.0 * r));
                        for (uint64_t k = 0; k < latFloats; k++) {
                            float x0d = c0 * x0v[k] - c1 * x0Prev[k];
                            xNext[k] = (sigmaDown / sigma) * lat[k] - std::expm1(-h) * x0d;
                        }
                    }
                    for (uint64_t k = 0; k < latFloats; k++) xNext[k] += nd(rng) * sigmaUp;
                    x0Prev = x0v;
                    tPrev = tNow;
                    havePrev = true;
                    break;
                }
                case SamplerKind::DDIM: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    for (uint64_t k = 0; k < latFloats; k++)
                        xNext[k] = x0v[k] + (sigmaNext / sigma) * (lat[k] - x0v[k]);
                    break;
                }
                case SamplerKind::Euler:
                default: {
                    if (last) {
                        std::copy(x0v.begin(), x0v.end(), xNext.begin());
                        break;
                    }
                    for (uint64_t k = 0; k < latFloats; k++)
                        xNext[k] = lat[k] + (lat[k] - x0v[k]) / sigma * dt;
                    break;
                }
            }

            double sx = 0, sx0 = 0;
            for (uint64_t k = 0; k < latFloats; k++) {
                lat[k] = xNext[k];
                sx += (double)xNext[k] * xNext[k];
                sx0 += (double)x0v[k] * x0v[k];
            }
            std::printf("step %2d/%d  t=%6.1f  sigma %.5f  |x| %.4f  |x0| %.4f  |e| %.4f  thr %.3f  median %.0f ms\n",
                        i + 1, steps, t, sigma, std::sqrt(sx / latFloats), std::sqrt(sx0 / latFloats),
                        std::sqrt(se / latFloats), thr, stepMsSum / (i + 1));
            report(onProgress, Progress::Kind::Step, i + 1, steps, 0.15 + 0.7 * (double)(i + 1) / steps, stepMs,
                   "step " + std::to_string(i + 1) + "/" + std::to_string(steps));
        }
        G.eng.profileDump();
        res.secPerStep = stepMsSum / 1000.0 / std::max(1, steps);

        // ---------------- hires refine: latent upscale + tiled img2img re-denoise
        // Plain bicubic on the decoded image adds no detail. Re-denoising the upscaled latent in
        // overlapping tiles lets the UNet paint real detail back in, and the tiles keep the work
        // inside what a 6 GB card can hold.
        if (p.hires > 1.01f && p.hiresRefine > 0.01f) {
            uint32_t rh = std::max(8u, (uint32_t)std::lround((double)Lh * p.hires));
            uint32_t rw = std::max(8u, (uint32_t)std::lround((double)Lw * p.hires));
            rh = (rh + 7u) & ~7u;
            rw = (rw + 7u) & ~7u;
            std::vector<float> rlat = upscaleLatentBilinear(lat, Lh, Lw, rh, rw);
            Euler rs = buildPartial(full, steps, p.hiresRefine, p.zsnr);
            const int rn = (int)rs.timesteps.size();
            // The refine pass reuses the step arena that was sized for the base canvas, so a tile
            // must never be larger than the base latent (96 px of canvas at 768 -> ~1.5 GB already).
            const uint32_t tile = std::min<uint32_t>(96u, std::min(Lh, Lw));
            const uint32_t ov = std::min<uint32_t>(24u, tile / 2);
            const uint32_t st = tile - ov;
            std::vector<float> acc((size_t)4 * rh * rw, 0.0f), wsum((size_t)rh * rw, 0.0f);
            std::printf("hires refine: %ux%u -> %ux%u  denoise %.2f  %d steps  tile %u\n", Lw, Lh, rw, rh,
                        p.hiresRefine, rn, tile);
            auto runPassHW = [&](const BufRef& pool, const BufRef& ctx, float t, uint32_t hh, uint32_t ww,
                                 const BufRef& xb, std::vector<float>& epsOut) {
                auto te = timeEmbHost(t, 320);
                G.eng.uploadNow(embHost, te.data(), te.size() * 4);
                G.eng.beginStep();
                BufRef temb = buildTemb(pool, addTimeB, embHost);
                BufRef eps = unetForward(xb, temb, ctx, hh, ww);
                G.eng.endStep();
                G.eng.waitStep();
                G.eng.downloadNow(eps, epsOut.data(), epsOut.size() * 4);
            };
            uint64_t tileCount = 0;
            for (uint32_t y0 = 0; y0 < rh;) {
                if (p.shouldCancel && p.shouldCancel()) fail("cancelled");
                uint32_t yy = (rh > tile) ? std::min(y0, rh - tile) : 0u;
                uint32_t y1 = std::min(rh, yy + tile);
                for (uint32_t x0 = 0; x0 < rw;) {
                    if (p.shouldCancel && p.shouldCancel()) fail("cancelled");
                    uint32_t xx = (rw > tile) ? std::min(x0, rw - tile) : 0u;
                    uint32_t x1 = std::min(rw, xx + tile);
                    uint32_t th = y1 - yy, tw = x1 - xx;
                    uint64_t tf = (uint64_t)4 * th * tw;
                    std::vector<float> blk(tf), epsC(tf), epsU(tf), resv(tf), x0v(tf), scaled(tf);
                    for (uint32_t c = 0; c < 4; c++)
                        for (uint32_t y = 0; y < th; y++)
                            std::memcpy(&blk[((size_t)c * th + y) * tw],
                                        &rlat[((size_t)c * rh + yy + y) * rw + xx], tw * 4);
                    uint32_t ts = p.seed * 2654435761u + yy * 73856093u + xx * 19349663u;
                    std::mt19937 rng(ts);
                    std::normal_distribution<float> nd(0.0f, 1.0f);
                    for (uint64_t k = 0; k < tf; k++) blk[k] += nd(rng) * rs.sigmas[0];
                    BufRef xb = G.scratchArena.alloc(tf * 4);
                    for (int i = 0; i < rn; i++) {
                        float t = rs.timesteps[i], sigma = rs.sigmas[i];
                        float inv = 1.0f / std::sqrt(sigma * sigma + 1.0f);
                        for (uint64_t k = 0; k < tf; k++) scaled[k] = blk[k] * inv;
                        G.eng.uploadNow(xb, scaled.data(), tf * 4);
                        runPassHW(poolCB, ctxC, t, th, tw, xb, epsC);
                        runPassHW(poolUB, ctxU, t, th, tw, xb, epsU);
                        for (uint64_t k = 0; k < tf; k++) resv[k] = epsU[k] + p.cfg * (epsC[k] - epsU[k]);
                        for (uint64_t k = 0; k < tf; k++) {
                            float xv = blk[k], e = resv[k];
                            float s2 = sigma * sigma + 1.0f;
                            x0v[k] = p.vpred ? e * (-sigma / std::sqrt(s2)) + xv / s2 : xv - sigma * e;
                        }
                        float dt = rs.sigmas[i + 1] - sigma;
                        for (uint64_t k = 0; k < tf; k++) blk[k] += (blk[k] - x0v[k]) / sigma * dt;
                    }
                    for (uint32_t c = 0; c < 4; c++)
                        for (uint32_t y = 0; y < th; y++) {
                            float wy = 1.0f;
                            if (yy > 0) wy = std::min(wy, (y + 0.5f) / (float)ov);
                            if (y1 < rh) wy = std::min(wy, (th - y - 0.5f) / (float)ov);
                            for (uint32_t bx = 0; bx < tw; bx++) {
                                float wx = 1.0f;
                                if (xx > 0) wx = std::min(wx, (bx + 0.5f) / (float)ov);
                                if (x1 < rw) wx = std::min(wx, (tw - bx - 0.5f) / (float)ov);
                                float w = wx * wy;
                                acc[((size_t)c * rh + yy + y) * rw + xx + bx] +=
                                    blk[((size_t)c * th + y) * tw + bx] * w;
                                if (c == 0) wsum[(size_t)(yy + y) * rw + xx + bx] += w;
                            }
                        }
                    tileCount++;
                    if (x1 >= rw) break;
                    x0 = xx + st;
                }
                if (y1 >= rh) break;
                y0 = yy + st;
            }
            for (uint32_t c = 0; c < 4; c++)
                for (uint32_t y = 0; y < rh; y++)
                    for (uint32_t x = 0; x < rw; x++) {
                        size_t i = ((size_t)c * rh + y) * rw + x;
                        acc[i] /= std::max(wsum[(size_t)y * rw + x], 1e-6f);
                    }
            lat.swap(acc);
            Lh = rh;
            Lw = rw;
            std::printf("hires refine: %llu tiles done, latent %ux%u\n", (unsigned long long)tileCount, Lw, Lh);
        }

        if (p.saveLatent && !p.latentOut.empty()) {
            if (writeLatent(p.latentOut, lat, Lh, Lw)) std::printf("latent saved: %s\n", p.latentOut.c_str());
        }
        if (p.noVae) {
            res.ok = true;
            res.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
            return res;
        }

        // ---------------- VAE decode
        if (p.shouldCancel && p.shouldCancel()) fail("cancelled");
        report(onProgress, Progress::Kind::Decode, steps, steps, 0.88, 0.0, "vae decode");
        auto tD0 = std::chrono::steady_clock::now();
        std::vector<float> img = vaeDecode(p, lat, Lh, Lw);
        uint32_t ow = Lw * 8, oh = Lh * 8;
        if (p.hires > 1.01f && p.hiresRefine <= 0.01f) upscaleImageBicubic(img, ow, oh, p.hires);
        res.vaeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tD0).count();
        std::string info = p.infoOut ? ofc::sdxl::infoText(p, m.modelName, gDeviceName) : std::string();
        if (!writeImageFile(p.outPath, img, ow, oh, p.pngCompress, info))
            fail("cannot write " + p.outPath);
        res.outPath = p.outPath;
        width = ow;
        height = oh;
        res.image.width = width;
        res.image.height = height;
        res.width = width;
        res.height = height;
        res.image.rgb.resize((size_t)3 * width * height);
        for (uint64_t k = 0; k < res.image.rgb.size(); k++) res.image.rgb[k] = img[k] * 0.5f + 0.5f;
        res.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
        res.ok = true;
        std::printf("wrote %s (%ux%u)%s\n", p.outPath.c_str(), width, height, seededFromImage ? " [img2img]" : "");
        report(onProgress, Progress::Kind::Done, steps, steps, 1.0, res.totalMs, "done");
        return res;
    } catch (const std::exception& e) {
        res.ok = false;
        res.error = e.what();
        std::printf("FAIL: %s\n", e.what());
        res.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
        return res;
    }
}

bool writePngFile(const std::string& path, const std::vector<float>& rgbChw, uint32_t w, uint32_t h, int compress,
                  const std::string& info) {
    return writeImageFile(path, rgbChw, w, h, compress, info);
}

RunResult decodeLatentFile(EvalModel& m, const GenParams& in, const std::string& latentPath,
                           const ProgressFn& onProgress) {
    RunResult res;
    auto tStart = std::chrono::steady_clock::now();
    std::string err;
    GenParams p = in;
    p.noVae = false;
    try {
        // --decode never touches the UNet, but it still needs the VAE path filled in.
        {
            ModelSet set;
            if (!ofc::sdxl::resolveModels(p, set, err)) fail(err);
        }
        if (!ensureDevice(p.spvDir, err, p.device)) fail(err);
        std::vector<float> lat;
        uint32_t lh = 0, lw = 0;
        if (!readLatent(latentPath, lat, lh, lw, err)) fail(err);
        report(onProgress, Progress::Kind::Load, 0, 1, 0.05, 0.0, "decoding latent " + latentPath);
        auto tv = std::chrono::steady_clock::now();
        std::vector<float> img = vaeDecode(p, lat, lh, lw);
        uint32_t ow = lw * 8, oh = lh * 8;
        if (p.hires > 1.01f) upscaleImageBicubic(img, ow, oh, p.hires);
        res.vaeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        report(onProgress, Progress::Kind::Decode, 1, 1, 0.9, res.vaeMs, "vae decoded");
        std::string out = p.outPath;
        if (out.empty()) out = latentPath + ".png";
        p.width = ow;
        p.height = oh;
        std::string info = p.infoOut ? infoText(p, "latent", m.deviceName) : std::string();
        if (!writeImageFile(out, img, ow, oh, p.pngCompress, info)) fail("cannot write " + out);
        res.image.width = ow;
        res.image.height = oh;
        res.image.rgb.resize((size_t)3 * res.image.width * res.image.height);
        for (uint64_t k = 0; k < res.image.rgb.size(); k++) res.image.rgb[k] = img[k] * 0.5f + 0.5f;
        res.outPath = out;
        res.width = res.image.width;
        res.height = res.image.height;
        res.device = m.deviceName;
        res.modelName = "latent";
        res.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
        res.ok = true;
        report(onProgress, Progress::Kind::Done, 1, 1, 1.0, res.totalMs, "done");
        return res;
    } catch (const std::exception& e) {
        res.ok = false;
        res.error = e.what();
        res.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
        return res;
    }
}

std::string deviceCheck(const std::string& spvDir, bool bench) {
    std::string err;
    if (!ensureDevice(spvDir, err, 0)) {
        return "device init failed: " + err + "\n(pass --spv <folder with compiled *.spv shaders>)\n";
    }
    std::string out = G.eng.deviceReport();
    const uint64_t mb = G.eng.deviceLocalMb();
    const uint64_t usable = mb > 1500 ? mb - 1500 : mb;  // the desktop holds part of the heap
    char mline[128];
    std::snprintf(mline, sizeof mline, "  estimate: %.0f MB usable after the desktop\n", (double)usable);
    out += "\ncanvas suggestions:\n";
    out += mline;
    out += "  SDXL: 768x768, 28 steps, --cfg 3.0 --spacing trailing --vpred, real negative prompt\n";
    if (usable < 5500) {
        out += "  ~6 GB class: close browser/GPU apps while generating; 832x1216 needs --scratch-mb <= 700\n";
    } else if (usable < 9500) {
        out += "  ~6-8 GB class: 768x768 solid; 832x1216 tight (close browser tabs, --scratch-mb <= 700)\n";
    } else {
        out += "  10 GB+: FLUX-class canvases fit in memory, but that architecture is not implemented yet\n";
    }
    out += "  arenas auto-size per canvas (--step-mb/--vstep-mb act as floors); OFC_KERNEL_PROFILE=1 prints kernel times\n";
    if (!bench) return out;

    auto now = [] { return std::chrono::steady_clock::now(); };
    auto msOf = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    // Persistent bench buffers live in the scratch arena: beginStep() resets the step arena.
    G.eng.initArenas(64ull << 20, 384ull << 20);
    auto salloc = [](uint64_t bytes) { return G.eng.scratchArena().alloc((bytes + 255) & ~255ull); };
    char line[256];
    out += "\nbench:\n";

    const uint64_t bwBytes = 64ull << 20;
    std::vector<float> host(bwBytes / 4);
    for (size_t i = 0; i < host.size(); i++) host[i] = (float)(i % 97) * 0.01f;
    BufRef buf = salloc(bwBytes);
    G.eng.uploadNow(buf, host.data(), bwBytes);  // warm-up
    auto t0 = now();
    G.eng.uploadNow(buf, host.data(), bwBytes);
    auto t1 = now();
    std::snprintf(line, sizeof line, "  upload %.0f MB: %.0f MB/s\n", bwBytes / 1048576.0,
                  bwBytes / 1048576.0 / (msOf(t0, t1) / 1000.0));
    out += line;
    std::vector<float> back(host.size());
    t0 = now();
    G.eng.downloadNow(buf, back.data(), bwBytes);
    t1 = now();
    std::snprintf(line, sizeof line, "  download %.0f MB: %.0f MB/s\n", bwBytes / 1048576.0,
                  bwBytes / 1048576.0 / (msOf(t0, t1) / 1000.0));
    out += line;

    const uint32_t M = 1024, N = 1024, K = 1024;
    std::vector<float> a((size_t)M * K);
    for (size_t i = 0; i < a.size(); i++) a[i] = (float)((i * 37) % 61) / 61.0f;
    std::vector<int8_t> q((size_t)N * K);
    std::vector<float> qs((size_t)N * K / 32);
    for (size_t i = 0; i < q.size(); i++) q[i] = (int8_t)((int)((i * 13) % 255) - 127);
    for (size_t i = 0; i < qs.size(); i++) qs[i] = 0.003f + 0.0001f * (float)(i % 7);
    BufRef bA = salloc((uint64_t)M * K * 4), bB = salloc((uint64_t)N * K * 4);
    BufRef bC = salloc((uint64_t)M * N * 4);
    BufRef bQ = salloc((uint64_t)N * K), bS = salloc((uint64_t)N * K / 32 * 4);
    G.eng.uploadNow(bA, a.data(), a.size() * 4);
    G.eng.uploadNow(bB, a.data(), a.size() * 4);
    G.eng.uploadNow(bQ, q.data(), q.size());
    G.eng.uploadNow(bS, qs.data(), qs.size() * 4);
    const int iters = 5;
    auto benchGemm = [&](bool q8) {
        G.eng.beginStep();
        for (int i = 0; i < iters; i++) {
            if (q8) gemmNtQ8(bA, bQ, bC, bS, M, N, K);
            else gemmNt(bA, bB, bC, M, N, K);
        }
        auto s0 = now();
        G.eng.endStep();
        G.eng.waitStep();
        return msOf(s0, now());
    };
    benchGemm(true);  // pipeline warm-up
    const double q8ms = benchGemm(true);
    const double f32ms = benchGemm(false);
    const double flops = 2.0 * (double)M * N * K * iters;
    std::snprintf(line, sizeof line, "  q8 gemm %ux%ux%u x%d: %.1f ms  %.0f GFLOPS\n", M, N, K, iters, q8ms,
                  flops / (q8ms / 1000.0) / 1e9);
    out += line;
    std::snprintf(line, sizeof line, "  fp32 gemm %ux%ux%u x%d: %.1f ms  %.0f GFLOPS\n", M, N, K, iters, f32ms,
                  flops / (f32ms / 1000.0) / 1e9);
    out += line;
    const double q8g = flops / (q8ms / 1000.0) / 1e9;
    if (q8g < 600) out += "  note: Q8 GEMM is far below this card's peak; check the driver state and OFC_KERNEL_PROFILE\n";
    return out;
}

}  // namespace sdxl
}  // namespace ofc
