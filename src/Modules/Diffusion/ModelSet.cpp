// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Segmented Fish
// sdvk - see LICENSE at the repository root. No warranty.
#include "ModelSet.hpp"

#include "Modules/Diffusion/Safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

namespace ofc {
namespace sdxl {

static std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static bool endsWith(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::string shortName(const std::string& path) {
    if (path.empty()) return "";
    std::string s = fs::path(path).filename().string();
    std::string l = lower(s);
    for (const char* e : {".safetensors", ".gguf", ".ckpt"}) {
        if (endsWith(l, e)) {
            s = s.substr(0, s.size() - std::strlen(e));
            break;
        }
    }
    return s;
}

enum : int { kUnknown = 0, kUnet = 1, kClipL = 2, kClipG = 3, kVae = 4, kFull = 5, kTaesd = 6 };

// Header-based classification. Works for diffusers component files, full checkpoints and
// taesd; the name is only a tie breaker (a full checkpoint also contains UNet keys).
static int classifyFile(const std::string& path, const std::string& leaf) {
    std::string l = lower(leaf);
    if (l.find("taesd") != std::string::npos || l.find("tiny_autoencoder") != std::string::npos) return kTaesd;
    if (endsWith(l, ".gguf")) return kUnet;
    Safetensors st;
    if (!st.open(path)) return kUnknown;
    bool teL = false, teG = false, unet = false, vae = false, full = false;
    for (auto& t : st.tensors()) {
        const std::string& n = t.name;
        if (n.rfind("model.diffusion_model.", 0) == 0) full = true;
        if (n.rfind("first_stage_model.", 0) == 0) full = full || n.find("decoder.") != std::string::npos;
        if (n.rfind("cond_stage_model.", 0) == 0) full = true;
        if (n.rfind("input_blocks.", 0) == 0 || n.rfind("middle_block.", 0) == 0 ||
            n.rfind("output_blocks.", 0) == 0 || n.rfind("unet.", 0) == 0 || n.rfind("down_blocks.", 0) == 0)
            unet = true;
        if (n.rfind("decoder.", 0) == 0 || n.rfind("vae.", 0) == 0) vae = true;
        if (n.rfind("text_model.", 0) == 0 || n.rfind("text_encoder.", 0) == 0 ||
            n.rfind("embeddings.token_embedding.", 0) == 0) {
            bool g = false;
            for (auto d : t.shape)
                if (d == 1280) g = true;
            if (g)
                teG = true;
            else
                teL = true;
        }
        if (n.rfind("text_encoder_2.", 0) == 0 || n.rfind("conditioner.embedders.1.", 0) == 0) teG = true;
    }
    if (full) return kFull;
    if (unet && !vae) return kUnet;
    if (vae && !unet && !teL) return kVae;
    if (teG) return kClipG;
    if (teL) return kClipL;
    return kUnknown;
}

static const char* clsName(int c) {
    switch (c) {
        case kUnet:
            return "UNet";
        case kClipL:
            return "CLIP-L";
        case kClipG:
            return "CLIP-G";
        case kVae:
            return "VAE";
        case kFull:
            return "full checkpoint";
        case kTaesd:
            return "TAESD";
    }
    return "unrecognized file";
}

struct Found {
    int cls;
    std::string path;
};

static std::vector<Found> scan(const std::string& dir, std::vector<std::string>& warn) {
    std::vector<Found> out;
    if (dir.empty()) return out;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        warn.push_back("model dir is not a directory: " + dir);
        return out;
    }
    std::vector<fs::path> files;
    // depth-first, sorted: a folder tree yields its components in a stable, predictable order
    for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
         it != end && !ec; it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        std::string leaf = it->path().filename().string();
        std::string l = lower(leaf);
        if (!(endsWith(l, ".safetensors") || endsWith(l, ".gguf"))) continue;
        files.push_back(it->path());
    }
    std::sort(files.begin(), files.end());
    for (auto& f : files) {
        int c = classifyFile(f.string(), f.filename().string());
        if (c == kUnknown) {
            warn.push_back("skip unrecognized model file: " + f.string());
            continue;
        }
        out.push_back({c, f.string()});
    }
    return out;
}


// Preference within a class, used when a folder holds duplicated copies of the same component
// (flat exports next to the original diffusers tree). Higher wins; 0 means "never auto-pick".
static int fitScore(const Found& f, const std::string& leaf) {
    std::string l = lower(leaf);
    std::string parent = lower(fs::path(f.path).parent_path().filename().string());
    bool inSubdir = !parent.empty() && parent != lower(fs::path(f.path).stem().string());
    int sc = 1;
    if (f.cls == kUnet) {
        if (endsWith(l, ".gguf")) sc += 40;                      // quantized, what the engine wants
        if (l.find("fp16") != std::string::npos) sc += 4;         // otherwise prefer half precision
        if (l.find("fp32") != std::string::npos || l.find("f32") != std::string::npos) sc -= 4;
    } else if (f.cls == kVae) {
        if (l.find("fp16") != std::string::npos) sc += 4;
    }
    if (l.find("config") != std::string::npos) sc -= 20;
    (void)inSubdir;
    return sc;
}

bool inspectUnetFile(const std::string& path, std::string& err) {
    if (endsWith(lower(path), ".gguf")) return true;
    int c = classifyFile(path, fs::path(path).filename().string());
    if (c == kUnet || c == kFull) return true;
    err = "not a usable UNet: " + path + " (" + clsName(c) + ")";
    return false;
}

static bool exists(const std::string& p) {
    if (p.empty()) return false;
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

static std::string join(const std::string& dir, const char* rel) {
    if (dir.empty()) return "";
    return (fs::path(dir) / rel).string();
}

// diffusers component layout, checked in order; the first hit wins
static bool fromDir(std::string& dst, const std::string& dir, std::initializer_list<const char*> rels) {
    if (!dst.empty() || dir.empty()) return !dst.empty();
    for (auto r : rels) {
        std::string p = join(dir, r);
        if (exists(p)) {
            dst = p;
            return true;
        }
    }
    return false;
}

// First existing path among the usual names for one component.
static std::string findByNames(const std::vector<Found>& found, int cls, std::initializer_list<const char*> names) {
    for (auto n : names) {
        for (auto& f : found)
            if (f.cls == cls && lower(fs::path(f.path).filename().string()) == lower(n)) return f.path;
    }
    return "";
}

bool resolveModels(GenParams& p, ModelSet& out, std::string& err) {
    ModelPaths& m = p.models;
    const std::string& dir = m.modelDir;

    // One cheap pass up front: every later decision is made from this index, so a flat
    // export ("unet_fp16.safetensors", "vae.safetensors") resolves exactly like a
    // diffusers tree ("unet/model.safetensors", "vae/diffusion_pytorch_model.safetensors").
    std::vector<Found> found = scan(dir, out.warn);

    // diffusers layout first: it is the canonical one and needs no guessing
    fromDir(m.unet, dir, {"unet/model.safetensors", "unet/diffusion_pytorch_model.safetensors"});
    fromDir(m.te1, dir, {"text_encoder/model.safetensors", "text_encoder/model.fp16.safetensors"});
    fromDir(m.te2, dir, {"text_encoder_2/model.safetensors", "text_encoder_2/model.fp16.safetensors"});
    fromDir(m.vae, dir, {"vae/model.safetensors", "vae/diffusion_pytorch_model.safetensors"});
    fromDir(m.taesd, dir, {"taesd/model.safetensors", "taesd/diffusion_pytorch_model.safetensors"});
    // GGUF wins for the UNet when present: it is the only form that keeps the weights quantized.
    if (m.unet.empty()) {
        for (auto& f : found)
            if (f.cls == kUnet && endsWith(lower(f.path), ".gguf")) {
                m.unet = f.path;
                break;
            }
    }
    // the "_<component>_model.safetensors" export convention
    if (m.unet.empty()) m.unet = findByNames(found, kUnet, {"unet_model.safetensors", "unet_fp16.safetensors"});
    if (m.te1.empty()) m.te1 = findByNames(found, kClipL, {"text_encoder_model.safetensors", "text_encoder.safetensors"});
    if (m.te2.empty()) m.te2 = findByNames(found, kClipG, {"text_encoder_2_model.safetensors", "text_encoder_2.safetensors"});
    if (m.vae.empty()) m.vae = findByNames(found, kVae, {"vae.safetensors", "vae_model.safetensors"});
    // tokenizers: the folder itself, or the vocab/merges pair sitting flat next to the weights
    if (!dir.empty()) {
        if (m.tok1.empty() && fs::is_directory(join(dir, "tokenizer"))) m.tok1 = join(dir, "tokenizer");
        if (m.tok2.empty() && fs::is_directory(join(dir, "tokenizer_2"))) m.tok2 = join(dir, "tokenizer_2");
        if (m.tok1.empty() && exists(join(dir, "vocab.json"))) m.tok1 = dir;
    }

    // pick the best-fitting copy per class; ties fall back to the first path in scan order
    std::vector<const Found*> first(kTaesd + 1, nullptr);
    std::vector<int> best(kTaesd + 1, -1);
    int count[kTaesd + 1] = {0};
    for (auto& f : found) {
        count[f.cls]++;
        int sc = fitScore(f, fs::path(f.path).filename().string());
        if (sc > best[f.cls]) {
            best[f.cls] = sc;
            first[f.cls] = &f;
        }
    }
    const char* label[kTaesd + 1] = {nullptr, "UNet", "CLIP-L", "CLIP-G", "VAE", nullptr, nullptr};
    for (int c = kUnet; c <= kVae; c++) {
        if (count[c] > 1 && first[c])
            out.warn.push_back(std::string("multiple ") + label[c] + " candidates, using " + first[c]->path);
    }
    if (m.te1.empty() && count[kClipL]) m.te1 = first[kClipL]->path;
    if (m.te2.empty() && count[kClipG]) m.te2 = first[kClipG]->path;
    if (m.vae.empty() && count[kVae]) m.vae = first[kVae]->path;
    if (m.taesd.empty() && count[kTaesd]) m.taesd = first[kTaesd]->path;

    // a full checkpoint can supply every component, including the ones the folder lacks
    const Found* full = first[kFull];
    if (m.unet.empty()) {
        if (const Found* f = first[kUnet])
            m.unet = f->path;
        else if (full)
            m.unet = full->path;
    }
    std::string e2;
    if (m.unet.empty()) {
        std::string q = dir.empty() ? " --model-dir <folder> or --unet <file.gguf|file.safetensors>" : "";
        err = "no UNet found;" + (q.empty() ? " " + dir + " has no unet/model.safetensors or *.gguf" : q);
        return false;
    }
    if (!inspectUnetFile(m.unet, e2)) {
        err = e2;
        return false;
    }
    bool fromFull = full && m.unet == full->path;
    if (fromFull) {
        if (m.te1.empty()) m.te1 = full->path;
        if (m.te2.empty()) m.te2 = full->path;
        if (m.vae.empty()) m.vae = full->path;
        out.warn.push_back("using single-file checkpoint: " + full->path +
                           " (text encoders and VAE are read from it)");
    }
    if (m.te1.empty() || m.te2.empty()) {
        err = std::string("missing text encoder (") + (m.te1.empty() ? "CLIP-L" : "CLIP-G") +
              "); pass --te1/--te2 or keep text_encoder/ and text_encoder_2/ next to the model";
        return false;
    }
    if (m.vae.empty()) {
        err = "missing VAE; pass --vae <file> or keep vae/ next to the model";
        return false;
    }
    // tokenizers can be skipped when the caller supplies ids or raw embeddings
    bool needTok = p.condTokens.empty() && p.embCondFile.empty();
    if (needTok) {
        if (m.tok1.empty() || m.tok2.empty()) {
            err = std::string("missing tokenizer (") + (m.tok1.empty() ? "vocab.json/merges.txt" : "tokenizer_2") +
                  "); pass --tok/--tok2, --model-dir, or feed raw embeddings with --emb-cond";
            return false;
        }
    }
    out.unetName = shortName(m.unet);
    return true;
}

}  // namespace sdxl
}  // namespace ofc
