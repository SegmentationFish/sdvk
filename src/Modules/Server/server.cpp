// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// sdvk-server: a tiny blocking HTTP/1.1 front end over the sdvk core.
// No third-party dependencies, no TLS, no thread pool. One request is served at a time,
// which is what a single GPU can actually do anyway.
//
//   GET  /                 web UI (served from webui/)
//   GET  /api/info         version, device, loaded model
//   GET  /api/models       classify a model folder (?dir=...)
//   POST /api/generate     run one job, returns JSON (or ?stream=1 for SSE progress)
//   GET  /api/image?path=  fetch a generated PNG from the output folder
//   POST /api/audio/process  run the audio node chain over a WAV in input|output
//   GET  /api/audio?path=  fetch a rendered WAV
//   GET  /api/job?id=      job status
//   POST /api/cancel       cancel the running job at the next step boundary
#include "Modules/Audio/Dsp.hpp"
#include "Modules/Audio/Wav.hpp"
#include "Modules/Diffusion/Eval.hpp"
#include "Modules/Diffusion/ModelSet.hpp"
#include "Modules/GPU/vk_engine.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace ofc::sdxl;

static const char* kVersion = "0.3.0";
static const char* kLicense = "GPL-3.0-only";

#ifdef _WIN32
using Sock = SOCKET;
static const Sock kBad = INVALID_SOCKET;
static void closeSock(Sock s) {
    // graceful close: without the shutdown the client can see the connection reset before the
    // last bytes are drained, which drops the body on loopback
    ::shutdown(s, SD_SEND);
    closesocket(s);
}
#else
using Sock = int;
static const Sock kBad = -1;
static void closeSock(Sock s) {
    ::shutdown(s, SHUT_WR);
    ::close(s);
}
#endif

// ---------------------------------------------------------------- small helpers

static std::string jsonEsc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                o += "\\\"";
                break;
            case '\\':
                o += "\\\\";
                break;
            case '\n':
                o += "\\n";
                break;
            case '\r':
                o += "\\r";
                break;
            case '\t':
                o += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char)c;
                }
        }
    }
    return o;
}

static std::string jstr(const std::string& s) { return "\"" + jsonEsc(s) + "\""; }

static std::string jnum(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

static std::string urlDecode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int a = hex(s[i + 1]), b = hex(s[i + 2]);
            if (a >= 0 && b >= 0) {
                o += (char)(a * 16 + b);
                i += 2;
                continue;
            }
        }
        o += s[i] == '+' ? ' ' : s[i];
    }
    return o;
}

static std::map<std::string, std::string> parseQuery(std::string q) {
    // req.query still carries the leading '?'; strip it or keys become "?async" and flags like
    // ?async=1 silently fall back to the blocking synchronous path.
    if (!q.empty() && q[0] == '?') q.erase(0, 1);
    std::map<std::string, std::string> m;
    size_t i = 0;
    while (i < q.size()) {
        size_t eq = q.find('=', i);
        size_t amp = q.find('&', i);
        if (amp == std::string::npos) amp = q.size();
        if (eq != std::string::npos && eq < amp) {
            m[urlDecode(q.substr(i, eq - i))] = urlDecode(q.substr(eq + 1, amp - eq - 1));
        } else {
            m[urlDecode(q.substr(i, amp - i))] = "";
        }
        i = amp + 1;
    }
    return m;
}

// Minimal JSON reader: enough for flat objects with string / number / bool values.
// A full parser would be nicer; this keeps the dependency list at zero.
struct FlatJson {
    std::map<std::string, std::string> s;
    std::map<std::string, double> n;
    std::map<std::string, bool> b;
    std::map<std::string, bool> isStr;
};

static bool parseFlatJson(const std::string& t, FlatJson& out) {
    size_t i = 0;
    auto ws = [&] {
        while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) i++;
    };
    auto str = [&](std::string& s) -> bool {
        if (i >= t.size() || t[i] != '"') return false;
        i++;
        s.clear();
        while (i < t.size() && t[i] != '"') {
            if (t[i] == '\\' && i + 1 < t.size()) {
                char c = t[i + 1];
                i += 2;
                switch (c) {
                    case 'n':
                        s += '\n';
                        break;
                    case 't':
                        s += '\t';
                        break;
                    case 'r':
                        s += '\r';
                        break;
                    case 'u': {
                        if (i + 4 > t.size()) return false;
                        unsigned v = (unsigned)std::strtoul(t.substr(i, 4).c_str(), nullptr, 16);
                        i += 4;
                        // UTF-16 escapes: handle the BMP subset, surrogate pairs fall back to '?'
                        if (v >= 0xD800 && v < 0xDC00 && i + 6 <= t.size() && t[i] == '\\' && t[i + 1] == 'u') {
                            unsigned lo = (unsigned)std::strtoul(t.substr(i + 2, 4).c_str(), nullptr, 16);
                            i += 6;
                            v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        if (v < 0x80) {
                            s += (char)v;
                        } else if (v < 0x800) {
                            s += (char)(0xC0 | (v >> 6));
                            s += (char)(0x80 | (v & 0x3F));
                        } else if (v < 0x10000) {
                            s += (char)(0xE0 | (v >> 12));
                            s += (char)(0x80 | ((v >> 6) & 0x3F));
                            s += (char)(0x80 | (v & 0x3F));
                        } else {
                            s += (char)(0xF0 | (v >> 18));
                            s += (char)(0x80 | ((v >> 12) & 0x3F));
                            s += (char)(0x80 | ((v >> 6) & 0x3F));
                            s += (char)(0x80 | (v & 0x3F));
                        }
                        break;
                    }
                    default:
                        s += c;
                }
                continue;
            }
            s += t[i++];
        }
        if (i >= t.size()) return false;
        i++;
        return true;
    };
    ws();
    if (i >= t.size() || t[i] != '{') return false;
    i++;
    while (i < t.size()) {
        ws();
        if (i < t.size() && t[i] == '}') {
            i++;
            break;
        }
        std::string key;
        if (!str(key)) return false;
        ws();
        if (i >= t.size() || t[i] != ':') return false;
        i++;
        ws();
        if (i < t.size() && t[i] == '"') {
            std::string v;
            if (!str(v)) return false;
            out.s[key] = v;
            out.isStr[key] = true;
        } else if (i < t.size() && t[i] == '[') {
            // arrays are not used by this API; skip the balanced region
            int depth = 0;
            while (i < t.size()) {
                if (t[i] == '[') depth++;
                if (t[i] == ']' && --depth == 0) {
                    i++;
                    break;
                }
                i++;
            }
        } else {
            size_t st = i;
            while (i < t.size() && t[i] != ',' && t[i] != '}') i++;
            std::string v = t.substr(st, i - st);
            while (!v.empty() && (v.back() == ' ' || v.back() == '\n' || v.back() == '\r')) v.pop_back();
            if (v == "true") {
                out.b[key] = true;
                out.n[key] = 1;
            } else if (v == "false") {
                out.b[key] = false;
                out.n[key] = 0;
            } else if (!v.empty() && v != "null") {
                out.n[key] = std::atof(v.c_str());
            }
        }
        ws();
        if (i < t.size() && t[i] == ',') i++;
    }
    return true;
}

static std::string readFile(const std::string& p) {
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return "";
    std::string s;
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

static std::string mimeOf(const std::string& p) {
    std::string e = fs::path(p).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    if (e == ".html" || e == ".htm") return "text/html; charset=utf-8";
    if (e == ".js") return "text/javascript; charset=utf-8";
    if (e == ".css") return "text/css; charset=utf-8";
    if (e == ".json") return "application/json; charset=utf-8";
    if (e == ".png") return "image/png";
    if (e == ".wav") return "audio/wav";
    if (e == ".ppm") return "image/x-portable-pixmap";
    if (e == ".svg") return "image/svg+xml";
    if (e == ".ico") return "image/x-icon";
    if (e == ".woff2") return "font/woff2";
    return "application/octet-stream";
}

// ---------------------------------------------------------------- job state

struct Job {
    std::string id;
    std::atomic<bool> running{false};
    std::atomic<bool> cancel{false};
    std::atomic<bool> done{false};
    std::atomic<int> step{0};
    int steps = 0;
    double progress = 0;
    std::string text, error, outPath;
    double totalMs = 0, secPerStep = 0;
    uint32_t seed = 0, w = 0, h = 0;
    std::mutex mu;
    std::vector<std::string> log;
};

static std::mutex gJobMu;
static std::map<std::string, std::shared_ptr<Job>> gJobs;
static std::string gLastError;
static EvalModel gModel;
static std::string gLoaded;
static std::mutex gModelMu;

static std::string newId() {
    static std::mt19937 rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%08x", rng());
    return buf;
}

// ---------------------------------------------------------------- request handling

struct Request {
    std::string method, path, query, body;
    std::map<std::string, std::string> headers;
};

struct Response {
    int code = 200;
    std::string type = "application/json; charset=utf-8";
    std::string body;
    std::vector<std::pair<std::string, std::string>> extra;
};

static void setJson(Response& r, int code, const std::string& body) {
    r.code = code;
    r.type = "application/json; charset=utf-8";
    r.body = body;
}

static void setErr(Response& r, int code, const std::string& msg) {
    setJson(r, code, "{\"ok\":false,\"error\":" + jstr(msg) + "}");
}

static void appendCors(Response& r) {
    r.extra.push_back({"Access-Control-Allow-Origin", "*"});
    r.extra.push_back({"Access-Control-Allow-Headers", "Content-Type"});
    r.extra.push_back({"Access-Control-Allow-Methods", "GET, POST, OPTIONS"});
}

static int gCurrentDevice = 0;
static std::string gWebRoot = "webui";
static std::string gOutDir = "output";
static std::string gSpv = "shaders";
static std::string gInputDir = "input";

// Turns a JSON body into GenParams. Unknown keys are ignored so the API stays forward compatible.
static bool paramsFromJson(const FlatJson& j, GenParams& p, std::string& err) {
    auto has = [&](const char* k) { return j.s.count(k) || j.n.count(k) || j.b.count(k); };
    if (has("prompt")) p.prompt = j.s.count("prompt") ? j.s.at("prompt") : "";
    if (!j.s.count("prompt") && !has("init")) {
        err = "missing \"prompt\"";
        return false;
    }
    if (has("negative")) p.negative = j.s.at("negative");
    if (has("model")) p.models.modelDir = j.s.at("model");
    if (has("unet")) p.models.unet = j.s.at("unet");
    if (has("te1")) p.models.te1 = j.s.at("te1");
    if (has("te2")) p.models.te2 = j.s.at("te2");
    if (has("vae")) p.models.vae = j.s.at("vae");
    if (has("tok")) p.models.tok1 = j.s.at("tok");
    if (has("tok2")) p.models.tok2 = j.s.at("tok2");
    if (j.n.count("device")) p.device = (int)j.n.at("device");
    if (j.n.count("steps")) p.steps = std::max(1, (int)j.n.at("steps"));
    if (j.n.count("cfg")) p.cfg = (float)j.n.at("cfg");
    if (j.n.count("rescale")) p.rescale = (float)j.n.at("rescale");
    if (j.n.count("eta")) p.eta = (float)j.n.at("eta");
    if (j.n.count("denoise")) p.denoise = (float)j.n.at("denoise");
    if (j.n.count("width")) p.width = (uint32_t)j.n.at("width");
    if (j.n.count("height")) p.height = (uint32_t)j.n.at("height");
    if (j.n.count("seed")) p.seed = (uint32_t)(int)j.n.at("seed");
    if (j.b.count("vpred")) p.vpred = j.b.at("vpred");
    if (j.b.count("zsnr")) p.zsnr = j.b.at("zsnr");
    if (j.b.count("dynthresh")) p.dynthresh = j.b.at("dynthresh");
    if (j.s.count("sampler") && !samplerFromName(j.s.at("sampler"), p.sampler)) {
        err = "unknown sampler: " + j.s.at("sampler");
        return false;
    }
    if (j.s.count("spacing")) p.spacing = j.s.at("spacing");
    if (j.n.count("hires")) p.hires = (float)j.n.at("hires");
    if (j.n.count("hiresDenoise")) p.hiresRefine = (float)j.n.at("hiresDenoise");
    if (j.n.count("vaeTile")) p.vaeTile = (uint32_t)j.n.at("vaeTile");
    if (j.n.count("vaeOverlap")) p.vaeOverlap = (uint32_t)j.n.at("vaeOverlap");
    if (j.n.count("vstepMb")) p.vStepMb = (uint32_t)j.n.at("vstepMb");
    if (j.n.count("paramMb")) p.paramMb = (uint32_t)j.n.at("paramMb");
    if (j.n.count("stepMb")) p.stepMb = (uint32_t)j.n.at("stepMb");
    if (j.n.count("scratchMb")) p.scratchMb = (uint32_t)j.n.at("scratchMb");
    if (j.s.count("lora")) p.lora = j.s.at("lora");
    if (j.n.count("loraScale")) p.loraScale = (float)j.n.at("loraScale");
    if (j.n.count("pngCompress")) p.pngCompress = (int)j.n.at("pngCompress");
    if (j.n.count("cropX")) p.cropOffsetX = (uint32_t)j.n.at("cropX");
    if (j.n.count("cropY")) p.cropOffsetY = (uint32_t)j.n.at("cropY");
    // img2img: the reference image is a path on this machine (POST /api/upload stores one and
    // returns its path). Every earlier version of this function accepted "init" in the has()
    // guard but never copied it into GenParams, so HTTP img2img silently did txt2img.
    if (j.s.count("init") && !j.s.at("init").empty()) {
        p.initImage = j.s.at("init");
        p.useInit = true;
    }
    // inpaint: "mask" is a path from POST /api/upload (white = repaint). Without it the same
    // request is plain img2img, so the two share every other field.
    if (j.s.count("mask") && !j.s.at("mask").empty()) {
        if (p.initImage.empty()) {
            err = "\"mask\" needs \"init\"";
            return false;
        }
        p.maskImage = j.s.at("mask");
    }
    if (j.n.count("maskFeather")) p.maskFeather = (uint32_t)std::max(0.0, j.n.at("maskFeather"));
    if (j.b.count("maskInvert")) p.maskInvert = j.b.at("maskInvert");
    if (j.n.count("cropW") && j.n.count("cropH")) {
        p.crop.x = j.n.count("cropX") ? (uint32_t)j.n.at("cropX") : 0;
        p.crop.y = j.n.count("cropY") ? (uint32_t)j.n.at("cropY") : 0;
        p.crop.w = (uint32_t)j.n.at("cropW");
        p.crop.h = (uint32_t)j.n.at("cropH");
    }
    if (p.height == 0) p.height = p.width;
    // SDXL/SD1.5 downsample the latent three times, so the pixel size must be a multiple of 64
    // (latent multiple of 8). A 720-wide canvas gives a 90-wide latent whose odd half breaks the
    // UNet skip shapes and only surfaces later as a confusing arena failure.
    if (p.width % 64 || p.height % 64 || p.width < 64 || p.height < 64) {
        err = "width/height must be multiples of 64 and at least 64 (e.g. 832x1216, 768x1024)";
        return false;
    }
    if (p.steps < 1 || p.steps > 200) {
        err = "steps must be in 1..200";
        return false;
    }
    return true;
}

static void applyDefaults(GenParams& p, const std::string& spv) {
    p.spvDir = spv;
    p.outPath = (fs::path(gOutDir) / ("sdvk_" + newId() + ".png")).string();
    p.saveLatent = false;
    p.latentOut.clear();
    p.infoOut = true;
    p.pngCompress = 1;
}

static void handleCancel(const std::map<std::string, std::string>& q, Response& r) {
    std::shared_ptr<Job> job;
    {
        std::lock_guard<std::mutex> lk(gJobMu);
        auto it = q.find("id");
        if (it == q.end()) {
            // no id: cancel whatever is running
            for (auto& kv : gJobs)
                if (kv.second->running) job = kv.second;
        } else {
            auto f = gJobs.find(it->second);
            if (f != gJobs.end()) job = f->second;
        }
    }
    if (!job) {
        setErr(r, 404, "no running job");
        return;
    }
    job->cancel = true;
    setJson(r, 200, "{\"ok\":true,\"id\":" + jstr(job->id) + ",\"cancel\":" + (job->running ? "\"requested\"" : "\"idle\"") +
                        "}");
}

static std::string jobJson(const std::shared_ptr<Job>& j) {
    std::ostringstream o;
    o << "{\"ok\":true,\"id\":" << jstr(j->id) << ",\"running\":" << (j->running ? "true" : "false")
      << ",\"done\":" << (j->done ? "true" : "false") << ",\"cancel\":" << (j->cancel ? "true" : "false")
      << ",\"step\":" << j->step.load()
      << ",\"steps\":" << j->steps << ",\"progress\":" << jnum(j->progress) << ",\"text\":" << jstr(j->text)
      << ",\"error\":" << jstr(j->error) << ",\"outPath\":" << jstr(j->outPath) << ",\"totalMs\":" << jnum(j->totalMs)
      << ",\"secPerStep\":" << jnum(j->secPerStep) << ",\"seed\":" << j->seed << ",\"width\":" << j->w
      << ",\"height\":" << j->h << "}";
    return o.str();
}

// Runs the job on the calling thread. Progress goes to the Job so a concurrent /api/job can read it.
static void runJob(const std::shared_ptr<Job>& job, GenParams p) {
    job->running = true;
    job->done = false;
    job->steps = p.steps;
    // The sampler polls this once per step, so cancel lands at the next step boundary instead of
    // killing the process mid-dispatch.
    p.shouldCancel = [job] { return job->cancel.load(); };
    ProgressFn onProgress = [job](const Progress& pr) {
        job->progress = pr.progress;
        job->text = pr.text;
        if (pr.kind == Progress::Kind::Step) job->step = pr.step;
        std::lock_guard<std::mutex> lk(job->mu);
        job->log.push_back(pr.text);
        if (job->log.size() > 200) job->log.erase(job->log.begin(), job->log.begin() + 100);
    };
    try {
        // The model stays resident between jobs; loading it is the expensive part.
        if (gLoaded != p.models.modelDir || gCurrentDevice != p.device) {
            unloadModel(gModel);
            std::string e;
            if (!loadModel(gModel, p, onProgress, e)) {
                job->error = e;
                job->done = true;
                job->running = false;
                return;
            }
            gLoaded = p.models.modelDir;
            gCurrentDevice = p.device;
        }
        RunResult r = run(gModel, p, onProgress);
        job->error = r.error;
        job->outPath = r.outPath;
        job->totalMs = r.totalMs;
        job->secPerStep = r.secPerStep;
        job->seed = r.seed;
        job->w = r.width;
        job->h = r.height;
    } catch (const std::exception& e) {
        job->error = e.what();
    }
    job->progress = 1.0;
    job->done = true;
    job->running = false;
}

static void handleInfo(Response& r) {
    std::ostringstream o;
    o << "{\"ok\":true,\"name\":\"sdvk\",\"version\":" << jstr(kVersion) << ",\"license\":" << jstr(kLicense)
      << ",\"device\":"
      << jstr(gModel.deviceName) << ",\"model\":" << jstr(gModel.modelName) << ",\"loaded\":"
      << (gModel.loaded ? "true" : "false") << ",\"modelDir\":" << jstr(gLoaded) << ",\"outDir\":" << jstr(gOutDir)
      << ",\"inputDir\":" << jstr(gInputDir)
      << ",\"caps\":[\"txt2img\",\"img2img\",\"inpaint\",\"upload\",\"vae\",\"hires\",\"batch\"],\"samplers\":["
         "\"euler\",\"euler_a\"],\"spacings\":[\"leading\",\"trailing\"]}";
    setJson(r, 200, o.str());
}

// Drop the resident model and its device memory, then hand the process working set back to the
// OS. Freeing VkDeviceMemory alone leaves pages resident in the process; on Windows the working
// set has to be trimmed explicitly for RAM to actually drop.
static void handleUnload(Response& r) {
    std::lock_guard<std::mutex> lk(gModelMu);
    unloadModel(gModel);
    gLoaded.clear();
#ifdef _WIN32
    SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
#endif
    std::ostringstream o;
    o << "{\"ok\":true,\"unloaded\":true}";
    setJson(r, 200, o.str());
}

static void handleModels(const std::map<std::string, std::string>& q, Response& r) {
    GenParams p;
    auto it = q.find("dir");
    p.models.modelDir = it == q.end() ? gLoaded : it->second;
    if (p.models.modelDir.empty()) {
        setErr(r, 400, "no model folder given (use ?dir=)");
        return;
    }
    ModelSet set;
    std::string err;
    p.prompt = "probe";
    if (!resolveModels(p, set, err)) {
        setErr(r, 400, err);
        return;
    }
    std::ostringstream o;
    o << "{\"ok\":true,\"dir\":" << jstr(p.models.modelDir) << ",\"unet\":" << jstr(p.models.unet) << ",\"te1\":"
      << jstr(p.models.te1) << ",\"te2\":" << jstr(p.models.te2) << ",\"vae\":" << jstr(p.models.vae) << ",\"tok1\":"
      << jstr(p.models.tok1) << ",\"tok2\":" << jstr(p.models.tok2) << ",\"name\":" << jstr(set.unetName)
      << ",\"warnings\":[";
    for (size_t i = 0; i < set.warn.size(); i++) o << (i ? "," : "") << jstr(set.warn[i]);
    o << "]}";
    setJson(r, 200, o.str());
}


static bool safeJoin(const std::string& root, const std::string& rel, std::string& out);

// GET /api/scan?dir=<folder>: list the components the direct-prompt UI can pick from.
// Paths are returned as absolute files so the client can round-trip them into /api/generate.
static void handleScan(const std::map<std::string, std::string>& q, Response& r) {
    auto it = q.find("dir");
    if (it == q.end() || it->second.empty()) {
        setErr(r, 400, "no model folder given (use ?dir=)");
        return;
    }
    const std::string dir = it->second;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        setErr(r, 400, "not a directory: " + dir);
        return;
    }
    std::vector<std::string> unets, checkpoints, vaes, clipL, clipG, loras;
    std::unordered_map<std::string, bool> seen;
    uint32_t scanned = 0;
    for (auto fit = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied);
         fit != fs::recursive_directory_iterator(); ++fit) {
        if (++scanned > 4000) break;
        if (!fit->is_regular_file(ec)) continue;
        std::string ext = fit->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (ext != ".gguf" && ext != ".safetensors") continue;
        std::string full = fit->path().string();
        if (!seen.emplace(full, true).second) continue;
        std::string rel = fs::relative(fit->path(), dir, ec).string();
        std::string low = rel;
        std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (low.find("lora") != std::string::npos) { loras.push_back(full); continue; }
        if (low.find("vae") != std::string::npos) { vaes.push_back(full); continue; }
        if (low.find("text_encoder_2") != std::string::npos || low.find("clip_g") != std::string::npos) {
            clipG.push_back(full);
            continue;
        }
        if (low.find("text_encoder") != std::string::npos || low.find("clip_l") != std::string::npos) {
            clipL.push_back(full);
            continue;
        }
        if (ext == ".gguf" || low.find("unet") != std::string::npos) { unets.push_back(full); continue; }
        std::error_code se;
        auto sz = fs::file_size(fit->path(), se);
        if (!se && sz > (1200ull << 20)) checkpoints.push_back(full);
        else unets.push_back(full);
    }
    auto sortv = [](std::vector<std::string>& v) { std::sort(v.begin(), v.end()); };
    sortv(unets); sortv(checkpoints); sortv(vaes); sortv(clipL); sortv(clipG); sortv(loras);

    GenParams p;
    p.models.modelDir = dir;
    p.prompt = "probe";
    ModelSet set;
    std::string err;
    std::string dUnet, dTe1, dTe2, dVae, dTok1, dTok2;
    if (resolveModels(p, set, err)) {
        dUnet = p.models.unet; dTe1 = p.models.te1; dTe2 = p.models.te2; dVae = p.models.vae;
        dTok1 = p.models.tok1; dTok2 = p.models.tok2;
    }
    auto arr = [&](const std::vector<std::string>& v) {
        std::ostringstream o;
        o << "[";
        for (size_t i = 0; i < v.size(); i++) { if (i) o << ","; o << jstr(v[i]); }
        o << "]";
        return o.str();
    };
    std::ostringstream o;
    o << "{\"ok\":true,\"dir\":" << jstr(dir) << ",\"unets\":" << arr(unets) << ",\"checkpoints\":"
      << arr(checkpoints) << ",\"vaes\":" << arr(vaes) << ",\"clipL\":" << arr(clipL) << ",\"clipG\":" << arr(clipG)
      << ",\"loras\":" << arr(loras) << ",\"defaults\":{\"unet\":" << jstr(dUnet) << ",\"te1\":" << jstr(dTe1)
      << ",\"te2\":" << jstr(dTe2) << ",\"vae\":" << jstr(dVae) << ",\"tok1\":" << jstr(dTok1)
      << ",\"tok2\":" << jstr(dTok2) << "},\"error\":" << jstr(err) << "}";
    setJson(r, 200, o.str());
}

// POST /api/decode {"latent":"input/x.bin","vae":".../vae.safetensors"}
// VAE-only: free the resident UNet first, decode the latent, write output/decode_<id>.png.
static void handleDecode(const Request& req, Response& r) {
    FlatJson j;
    if (!parseFlatJson(req.body, j) || !j.s.count("latent")) {
        setErr(r, 400, "expected {\"latent\":\"input/x.bin\",\"vae\":...}");
        return;
    }
    std::string base = fs::path(j.s.at("latent")).filename().string();
    std::string latent;
    {
        std::lock_guard<std::mutex> lk(gModelMu);
        unloadModel(gModel);
        gLoaded.clear();
    }
    std::error_code ec;
    if (!safeJoin(gInputDir, base, latent) || !fs::is_regular_file(latent, ec)) {
        if (!safeJoin(gOutDir, base, latent) || !fs::is_regular_file(latent, ec)) {
            setErr(r, 404, "latent not found (upload it first): " + base);
            return;
        }
    }
    GenParams p;
    applyDefaults(p, gSpv);
    p.models.modelDir = j.s.count("model") ? j.s.at("model") : gLoaded;
    if (j.s.count("vae")) p.models.vae = j.s.at("vae");
    if (j.n.count("hires")) p.hires = (float)j.n.at("hires");
    p.pngCompress = 1;
    p.infoOut = true;
    EvalModel m;
    RunResult res = decodeLatentFile(m, p, latent, nullptr);
    std::ostringstream o;
    o << "{\"ok\":" << (res.ok ? "true" : "false") << ",\"outPath\":" << jstr(res.outPath)
      << ",\"width\":" << res.width << ",\"height\":" << res.height << ",\"totalMs\":" << jnum(res.totalMs)
      << ",\"vaeMs\":" << jnum(res.vaeMs) << ",\"error\":" << jstr(res.error) << "}";
    setJson(r, res.ok ? 200 : 500, o.str());
}

static void handleGenerate(const Request& req, Response& r) {
    FlatJson j;
    if (!parseFlatJson(req.body, j)) {
        setErr(r, 400, "body is not a JSON object");
        return;
    }
    GenParams p;
    std::string err;
    if (!paramsFromJson(j, p, err)) {
        setErr(r, 400, err);
        return;
    }
    std::string spv = "shaders";
    if (j.s.count("spv")) spv = j.s.at("spv");
    applyDefaults(p, spv);

    auto job = std::make_shared<Job>();
    job->id = newId();
    {
        std::lock_guard<std::mutex> lk(gJobMu);
        gJobs[job->id] = job;
        if (gJobs.size() > 64) gJobs.erase(gJobs.begin());
    }

    // Request::query is never populated by the HTTP parser; the query string lives in req.path
    // after the '?'. Reading the empty field made every call take the blocking synchronous path.
    size_t qm = req.path.find('?');
    auto q = parseQuery(qm == std::string::npos ? std::string() : req.path.substr(qm + 1));
    bool async = q.count("async") != 0;
    if (async) {
        // Detached run, caller polls /api/job. The GPU is serial, so a second job queues here.
        std::thread([job, p] {
            std::lock_guard<std::mutex> lk(gModelMu);
            runJob(job, p);
        }).detach();
        setJson(r, 202, jobJson(job));
        return;
    }
    {
        std::lock_guard<std::mutex> lk(gModelMu);
        runJob(job, p);
    }
    std::ostringstream o;
    o << "{\"ok\":" << (job->error.empty() ? "true" : "false") << ",\"error\":" << jstr(job->error)
      << ",\"id\":" << jstr(job->id) << ",\"outPath\":" << jstr(job->outPath) << ",\"totalMs\":" << jnum(job->totalMs)
      << ",\"secPerStep\":" << jnum(job->secPerStep) << ",\"seed\":" << job->seed << ",\"width\":" << job->w
      << ",\"height\":" << job->h << ",\"model\":" << jstr(gModel.modelName) << "}";
    setJson(r, job->error.empty() ? 200 : 500, o.str());
}

static void handleJob(const std::map<std::string, std::string>& q, Response& r) {
    auto it = q.find("id");
    if (it == q.end()) {
        setErr(r, 400, "missing id");
        return;
    }
    std::shared_ptr<Job> job;
    {
        std::lock_guard<std::mutex> lk(gJobMu);
        auto f = gJobs.find(it->second);
        if (f != gJobs.end()) job = f->second;
    }
    if (!job) {
        setErr(r, 404, "no such job");
        return;
    }
    setJson(r, 200, jobJson(job));
}

static bool safeJoin(const std::string& root, const std::string& rel, std::string& out) {
    fs::path r = fs::weakly_canonical(root);
    fs::path f = fs::weakly_canonical(r / rel);
    std::string rs = r.string(), fs2 = f.string();
    if (fs2.size() < rs.size()) return false;
    for (auto& c : rs)
        if (c == '\\') c = '/';
    for (auto& c : fs2)
        if (c == '\\') c = '/';
    if (fs2.compare(0, rs.size(), rs) != 0) return false;
    out = f.string();
    return true;
}

static bool serveFile(const std::string& path, Response& r, bool inlineOk) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return false;
    r.type = mimeOf(path);
    r.body = readFile(path);
    if (r.body.empty()) return false;
    r.extra.push_back({"Cache-Control", inlineOk ? "no-cache" : "public, max-age=3600"});
    return true;
}

static void handleImage(const std::map<std::string, std::string>& q, Response& r) {
    auto it = q.find("path");
    if (it == q.end()) {
        setErr(r, 400, "missing path");
        return;
    }
    std::string full;
    // Only files under the output folder can be fetched, so the endpoint cannot be used to read the disk.
    // input/ is added for reads so the UI can show the reference image or mask it just uploaded.
    const std::string base = fs::path(it->second).filename().string();
    if (!safeJoin(gOutDir, base, full) || !serveFile(full, r, false)) {
        if (!safeJoin(gInputDir, base, full) || !serveFile(full, r, false)) setErr(r, 404, "no such image");
    }
}

static void handleDevices(Response& r) {
    std::ostringstream o;
    o << "[";
    auto list = ofc::gpu::vkListDevices();
    // caps tells the front end what a device can actually do. Every Vulkan device runs the full
    // pipeline; the CPU entry is an offload target for the memory-hungry tail (VAE encode/decode)
    // so a 6 GiB card does not have to hold the decoder arena on top of the UNet.
    const char* gpuCaps = "\"caps\":[\"txt2img\",\"img2img\",\"inpaint\",\"vae\",\"decode\"],\"backend\":\"vulkan\","
                          "\"full\":true";
    for (size_t i = 0; i < list.size(); i++) {
        // "index|name|mb"
        std::string x = list[i];
        size_t a1 = x.find('|'), a2 = x.find('|', a1 + 1);
        std::string idx = a1 == std::string::npos ? "0" : x.substr(0, a1);
        std::string name = a1 == std::string::npos ? x : x.substr(a1 + 1, a2 == std::string::npos ? std::string::npos : a2 - a1 - 1);
        std::string mb = a2 == std::string::npos ? "0" : x.substr(a2 + 1);
        if (i) o << ",";
        o << "{\"index\":" << idx << ",\"name\":" << jstr(name) << ",\"vramMb\":" << mb << "," << gpuCaps << "}";
    }
    // host RAM, so the UI can warn before a CPU-side decode
    double hostMb = 0;
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) hostMb = (double)ms.ullTotalPhys / 1048576.0;
#endif
    if (!list.empty()) o << ",";
    o << "{\"index\":-1,\"name\":\"CPU (host RAM)\",\"vramMb\":" << (long long)hostMb
      << ",\"backend\":\"cpu\",\"full\":false,\"caps\":[\"vae\",\"encode\",\"decode\",\"io\"]}";
    o << "]";
    setJson(r, 200, o.str());
}

// ---------------------------------------------------------------- image upload (img2img / inpaint)

static std::string b64Decode(const std::string& in) {
    static int8_t tbl[256];
    static bool built = false;
    if (!built) {
        for (int i = 0; i < 256; i++) tbl[i] = -1;
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) tbl[(uint8_t)a[i]] = (int8_t)i;
        built = true;
    }
    std::string out;
    out.reserve(in.size() * 3 / 4 + 8);
    int acc = 0, bits = 0;
    for (unsigned char c : in) {
        int8_t v = tbl[c];
        if (v < 0) continue;  // newlines, padding, the data: prefix is stripped by the caller
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char)((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// POST /api/upload {"name":"ref.png","data":"data:image/png;base64,..."}
// -> {"ok":true,"path":"input/xxx_ref.png"}. The path is what "init" wants on /api/generate.
static void handleUpload(const Request& req, Response& r) {
    FlatJson j;
    if (!parseFlatJson(req.body, j) || !j.s.count("data")) {
        setErr(r, 400, "expected {\"name\":..., \"data\":\"base64\"}");
        return;
    }
    std::string b64 = j.s.at("data");
    size_t comma = b64.find(',');
    if (comma != std::string::npos && b64.compare(0, 5, "data:") == 0) b64 = b64.substr(comma + 1);
    std::string raw = b64Decode(b64);
    if (raw.size() < 64) {
        setErr(r, 400, "empty image payload");
        return;
    }
    std::string name = j.s.count("name") ? j.s.at("name") : "upload.png";
    for (auto& c : name)
        if (!isalnum((unsigned char)c) && c != '.' && c != '_' && c != '-') c = '_';
    if (name.empty() || name[0] == '.') name = "upload.png";
    std::error_code ec;
    fs::create_directories(gInputDir, ec);
    std::string rel = (fs::path(gInputDir) / (newId() + "_" + name)).string();
    std::ofstream f(rel, std::ios::binary);
    if (!f) {
        setErr(r, 500, "cannot write " + rel);
        return;
    }
    f.write(raw.data(), (std::streamsize)raw.size());
    f.close();
    std::ostringstream o;
    o << "{\"ok\":true,\"path\":" << jstr(rel) << ",\"bytes\":" << raw.size() << "}";
    setJson(r, 200, o.str());
}


// ---------------------------------------------------------------- audio workflow

// The node graph reaches the server as one string because the request parser is deliberately
// flat: "op:key=value,key=value;op:key=value". Unknown keys are ignored, so the UI can send a
// superset of what an op reads without breaking older builds.
static bool parseAudioOps(const std::string& spec, std::vector<std::pair<std::string, std::map<std::string, double>>>& out,
                          std::string& err) {
    size_t i = 0;
    while (i < spec.size()) {
        size_t semi = spec.find(';', i);
        std::string item = spec.substr(i, semi == std::string::npos ? std::string::npos : semi - i);
        i = semi == std::string::npos ? spec.size() : semi + 1;
        // trim
        while (!item.empty() && (item.front() == ' ' || item.front() == '\t')) item.erase(item.begin());
        while (!item.empty() && (item.back() == ' ' || item.back() == '\t')) item.pop_back();
        if (item.empty()) continue;
        size_t colon = item.find(':');
        std::pair<std::string, std::map<std::string, double>> op;
        op.first = item.substr(0, colon == std::string::npos ? std::string::npos : colon);
        if (colon != std::string::npos) {
            size_t p = colon + 1;
            while (p <= item.size()) {
                size_t comma = item.find(',', p);
                std::string kv = item.substr(p, comma == std::string::npos ? std::string::npos : comma - p);
                size_t eq = kv.find('=');
                if (eq != std::string::npos) {
                    const std::string k = kv.substr(0, eq);
                    const std::string v = kv.substr(eq + 1);
                    try {
                        op.second[k] = std::stod(v);
                    } catch (const std::exception&) {
                        // a non-numeric value is a client bug; leave the default in place
                    }
                }
                if (comma == std::string::npos) break;
                p = comma + 1;
            }
        }
        if (op.first.empty()) {
            err = "empty audio op in spec";
            return false;
        }
        out.push_back(op);
    }
    return true;
}

// POST /api/audio/process
//   {"input":"input/a.wav","ops":"normalize-peak:targetDb=-1;resample:rate=48000",
//    "bits":24,"spectrogram":true,"specW":1024,"specH":384}
// -> {"ok":true,"path":"output/a_xxx.wav","seconds":..,"sampleRate":..,"channels":..,"peakDb":..,
//     "detail":[...],"spectrogram":"output/a_xxx_spec.png"}
static void handleAudioProcess(const Request& req, Response& r) {
    FlatJson j;
    if (!parseFlatJson(req.body, j) || !j.s.count("input")) {
        setErr(r, 400, "expected {\"input\":\"input/x.wav\",\"ops\":\"...\"}");
        return;
    }
    // Only the two spool folders are readable, same rule as /api/image.
    const std::string base = fs::path(j.s.at("input")).filename().string();
    std::string inPath;
    std::error_code ec;
    bool found = safeJoin(gInputDir, base, inPath) && fs::is_regular_file(inPath, ec);
    if (!found) found = safeJoin(gOutDir, base, inPath) && fs::is_regular_file(inPath, ec);
    if (!found) {
        setErr(r, 404, "no such audio file: " + base);
        return;
    }
    ofc::audio::Audio a;
    std::string err;
    if (!ofc::audio::readWav(inPath, a, err)) {
        setErr(r, 400, err);
        return;
    }
    std::vector<std::pair<std::string, std::map<std::string, double>>> ops;
    if (j.s.count("ops") && !parseAudioOps(j.s.at("ops"), ops, err)) {
        setErr(r, 400, err);
        return;
    }
    std::ostringstream detail;
    detail << "[";
    for (size_t i = 0; i < ops.size(); i++) {
        ofc::audio::OpArgs args;
        auto get = [&](const char* k, double dflt) {
            auto it = ops[i].second.find(k);
            return it == ops[i].second.end() ? dflt : it->second;
        };
        args.gainDb = get("gainDb", args.gainDb);
        args.targetDb = get("targetDb", args.targetDb);
        args.inSec = get("inSec", args.inSec);
        args.outSec = get("outSec", args.outSec);
        args.padSec = get("padSec", args.padSec);
        args.thresholdDb = get("thresholdDb", args.thresholdDb);
        args.seconds = get("seconds", args.seconds);
        args.mix = get("mix", args.mix);
        args.decay = get("decay", args.decay);
        args.curve = (int)get("curve", args.curve);
        args.rate = (uint32_t)std::max(0.0, get("rate", args.rate));
        args.seed = (uint32_t)std::max(0.0, get("seed", args.seed));
        std::string what;
        if (!ofc::audio::applyOp(a, ops[i].first, args, what, err)) {
            setErr(r, 400, err);
            return;
        }
        if (i) detail << ",";
        detail << "{\"op\":" << jstr(ops[i].first) << ",\"detail\":" << jstr(what) << "}";
    }
    detail << "]";
    int bits = 24;
    if (j.n.count("bits")) bits = (int)j.n.at("bits");
    if (bits != 16 && bits != 24 && bits != 32) bits = 24;
    fs::create_directories(gOutDir, ec);
    std::string id = newId();
    std::string outName = id + "_audio.wav";
    if (j.s.count("name")) {
        std::string n = fs::path(j.s.at("name")).stem().string();
        for (auto& c : n)
            if (!isalnum((unsigned char)c) && c != '_' && c != '-') c = '_';
        if (!n.empty()) outName = id + "_" + n + ".wav";
    }
    const std::string outPath = (fs::path(gOutDir) / outName).string();
    if (!ofc::audio::writeWav(outPath, a, bits, err)) {
        setErr(r, 500, err);
        return;
    }
    std::string specPath;
    bool wantSpec = j.b.count("spectrogram") ? j.b.at("spectrogram") : false;
    if (wantSpec) {
        uint32_t sw = 1024, sh = 384;
        if (j.n.count("specW")) sw = (uint32_t)std::max(64.0, j.n.at("specW"));
        if (j.n.count("specH")) sh = (uint32_t)std::max(64.0, j.n.at("specH"));
        auto s = ofc::audio::spectrogram(a, 2048, 512);
        auto img = ofc::audio::spectrogramImage(s, sw, sh);
        specPath = (fs::path(gOutDir) / (id + "_audio_spec.png")).string();
        if (!writePngFile(specPath, img, sw, sh, 1, "sdvk audio spectrogram")) specPath.clear();
    }
    std::ostringstream o;
    o << "{\"ok\":true,\"path\":" << jstr(outPath) << ",\"bits\":" << bits << ",\"sampleRate\":" << a.sampleRate
      << ",\"channels\":" << a.channels << ",\"frames\":" << (unsigned long long)a.frames()
      << ",\"seconds\":" << jnum(a.seconds()) << ",\"peakDb\":" << jnum(ofc::audio::peakDb(a))
      << ",\"rmsDb\":" << jnum(ofc::audio::rmsDb(a)) << ",\"detail\":" << detail.str();
    if (!specPath.empty()) o << ",\"spectrogram\":" << jstr(specPath);
    o << "}";
    setJson(r, 200, o.str());
}

// GET /api/audio?path=  — fetch a rendered WAV from the output folder (the UI plays it inline).
static void handleAudioFile(const std::map<std::string, std::string>& q, Response& r) {
    auto it = q.find("path");
    if (it == q.end()) {
        setErr(r, 400, "missing path");
        return;
    }
    std::string full;
    if (!safeJoin(gOutDir, fs::path(it->second).filename().string(), full) || !serveFile(full, r, false)) {
        setErr(r, 404, "no such audio file");
    }
}

static Response dispatch(const Request& req) {
    Response r;
    appendCors(r);
    if (req.method == "OPTIONS") {
        r.code = 204;
        return r;
    }
    size_t qm = req.path.find('?');
    std::string path = qm == std::string::npos ? req.path : req.path.substr(0, qm);
    std::string query = qm == std::string::npos ? "" : req.path.substr(qm + 1);
    auto q = parseQuery(query);

    if (path == "/api/info") {
        handleInfo(r);
    } else if (path == "/api/devices") {
        handleDevices(r);
    } else if (path == "/api/models") {
        handleModels(q, r);
    } else if (path == "/api/scan") {
        handleScan(q, r);
    } else if (path == "/api/decode" && req.method == "POST") {
        handleDecode(req, r);
    } else if (path == "/api/generate" && req.method == "POST") {
        handleGenerate(req, r);
    } else if (path == "/api/upload" && req.method == "POST") {
        handleUpload(req, r);
    } else if (path == "/api/unload" && req.method == "POST") {
        handleUnload(r);
    } else if (path == "/api/job") {
        handleJob(q, r);
    } else if (path == "/api/cancel") {
        handleCancel(q, r);
    } else if (path == "/api/image") {
        handleImage(q, r);
    } else if (path == "/api/audio/process" && req.method == "POST") {
        handleAudioProcess(req, r);
    } else if (path == "/api/audio") {
        handleAudioFile(q, r);
    } else if (path == "/api/health") {
        setJson(r, 200, "{\"ok\":true}");
    } else if (req.method == "GET") {
        std::string rel = path == "/" ? "index.html" : path.substr(1);
        std::string full;
        if (!safeJoin(gWebRoot, rel, full) || !serveFile(full, r, true)) {
            setErr(r, 404, "not found: " + path);
        }
    } else {
        setErr(r, 405, "method not allowed");
    }
    return r;
}

// ---------------------------------------------------------------- socket plumbing

static std::string statusText(int c) {
    switch (c) {
        case 200:
            return "OK";
        case 202:
            return "Accepted";
        case 204:
            return "No Content";
        case 400:
            return "Bad Request";
        case 404:
            return "Not Found";
        case 405:
            return "Method Not Allowed";
        case 413:
            return "Payload Too Large";
        case 500:
            return "Internal Server Error";
    }
    return "OK";
}

static bool sendAll(Sock s, const char* p, size_t n) {
    while (n) {
        int k = (int)::send(s, p, (int)n, 0);
        if (k <= 0) return false;
        p += k;
        n -= (size_t)k;
    }
    return true;
}

// Drain whatever the peer sent before we close, so the last response is not truncated by a RST.
static void drain(Sock s) {
#ifdef _WIN32
    char buf[1024];
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    for (int i = 0; i < 64 && ::recv(s, buf, sizeof(buf), 0) > 0; i++) {
    }
#endif
}

static bool readRequest(Sock s, Request& req) {
    std::string buf;
    char tmp[8192];
    size_t headerEnd = std::string::npos;
    while (headerEnd == std::string::npos) {
        int n = (int)::recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
        headerEnd = buf.find("\r\n\r\n");
        if (buf.size() > 1u << 20) return false;
    }
    std::string head = buf.substr(0, headerEnd);
    std::string rest = buf.substr(headerEnd + 4);
    std::istringstream hs(head);
    std::string line;
    std::getline(hs, line);
    {
        std::istringstream ls(line);
        std::string ver;
        ls >> req.method >> req.path >> ver;
    }
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t c = line.find(':');
        if (c == std::string::npos) continue;
        std::string k = line.substr(0, c), v = line.substr(c + 1);
        std::transform(k.begin(), k.end(), k.begin(), ::tolower);
        while (!v.empty() && (v[0] == ' ' || v[0] == '\t')) v.erase(v.begin());
        req.headers[k] = v;
    }
    size_t want = 0;
    auto cl = req.headers.find("content-length");
    if (cl != req.headers.end()) want = (size_t)std::strtoul(cl->second.c_str(), nullptr, 10);
    if (want > (64u << 20)) return false;
    req.body = rest;
    while (req.body.size() < want) {
        int n = (int)::recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        req.body.append(tmp, (size_t)n);
    }
    if (req.body.size() > want) req.body.resize(want);
    return true;
}

static void serveOne(Sock s) {
    Request req;
    if (!readRequest(s, req)) {
        closeSock(s);
        return;
    }
    Response r = dispatch(req);
    std::ostringstream head;
    head << "HTTP/1.1 " << r.code << " " << statusText(r.code) << "\r\n";
    head << "Content-Type: " << r.type << "\r\n";
    head << "Content-Length: " << r.body.size() << "\r\n";
    head << "Connection: close\r\n";
    for (auto& h : r.extra) head << h.first << ": " << h.second << "\r\n";
    head << "\r\n";
    std::string h = head.str();
    if (sendAll(s, h.data(), h.size()) && !r.body.empty()) sendAll(s, r.body.data(), r.body.size());
    drain(s);
    closeSock(s);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int port = 7500;
    std::string spv = "shaders";
    std::string modelDir;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--port" || a == "-p") {
            port = std::atoi(next().c_str());
        } else if (a == "--spv") {
            spv = next();
            gSpv = spv;
        } else if (a == "-m" || a == "--model-dir") {
            modelDir = next();
        } else if (a == "--web") {
            gWebRoot = next();
        } else if (a == "--out-dir") {
            gOutDir = next();
        } else if (a == "--version") {
            std::printf("sdvk-server %s\nlicense: %s\n", kVersion, kLicense);
            return 0;
        } else if (a == "--help" || a == "-h") {
            std::printf(
                "sdvk-server %s - HTTP API for the sdvk engine\n\n"
                "usage: sdvk-server [options]\n\n"
                "  -p, --port <n>        listen port (default 7500)\n"
                "  -m, --model-dir <dir>  load this model folder on first request if unset\n"
                "      --spv <dir>        compiled shader folder (default shaders)\n"
                "      --web <dir>        web UI folder (default webui)\n"
                "      --out-dir <dir>    generated images (default output)\n\n"
                "endpoints:\n"
                "  POST /api/generate    {\"prompt\":\"...\", ...}  one image per call\n"
                "  GET  /api/info        engine and model state\n"
                "  GET  /api/models      ?dir=<folder> classification\n"
                "  GET  /api/scan        ?dir=<folder> list unet/vae/clip/lora files\n"
                "  POST /api/decode      {latent, vae} VAE-only latent -> png\n"
                "  GET  /api/job         ?id=<job> progress of an async job\n"
                "  POST /api/cancel      ?id=<job> stop at the next sampler step\n"
                "  GET  /api/image       ?path=<file.png> fetch a generated image\n\n"
                "license: %s (free software, no warranty)\n",
                kVersion, kLicense);
            return 0;
        } else {
            std::printf("unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    std::error_code ec;
    fs::create_directories(gOutDir, ec);

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::printf("error: WSAStartup failed\n");
        return 1;
    }
#endif
    Sock srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv == kBad) {
        std::printf("error: socket() failed (%d)\n", (int)WSAGetLastError());
        return 1;
    }
    int reuse = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((unsigned short)port);
    if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) != 0) {
#ifdef _WIN32
        int e = WSAGetLastError();
        std::printf("error: bind 127.0.0.1:%d failed (%d)%s\n", port, e,
                    e == 10013 ? " - port is reserved by the OS, pick another one" : "");
#else
        std::printf("error: bind 127.0.0.1:%d failed\n", port);
#endif
        return 1;
    }
    if (::listen(srv, 8) != 0) {
        std::printf("error: listen failed\n");
        return 1;
    }
    std::printf("sdvk-server %s listening on http://127.0.0.1:%d\n", kVersion, port);
    std::printf("  web:    %s\n  models: %s\n  output: %s\n  shaders: %s\n", gWebRoot.c_str(),
                modelDir.empty() ? "(set per request)" : modelDir.c_str(), gOutDir.c_str(), spv.c_str());
    if (!modelDir.empty()) {
        GenParams p;
        p.models.modelDir = modelDir;
        p.spvDir = spv;
        p.prompt = "probe";
        ModelSet set;
        std::string e;
        if (resolveModels(p, set, e)) {
            std::printf("  model:  %s\n", set.unetName.c_str());
            gLoaded = modelDir;
            std::string le;
            if (!loadModel(gModel, p, nullptr, le)) std::printf("  load:   %s\n", le.c_str());
            else
                std::printf("  device: %s\n", gModel.deviceName.c_str());
        } else {
            std::printf("  model:  %s\n", e.c_str());
        }
    }
    for (;;) {
        Sock c = ::accept(srv, nullptr, nullptr);
        if (c == kBad) continue;
        serveOne(c);
    }
}
