const $ = (id) => document.getElementById(id);
const state = { running: false, job: null, timer: null };

function log(line) {
    const box = $("log");
    const d = document.createElement("div");
    const t = new Date().toLocaleTimeString([], { hour12: false });
    d.textContent = `${t}  ${line}`;
    box.appendChild(d);
    box.scrollTop = box.scrollHeight;
    while (box.childElementCount > 300) box.removeChild(box.firstChild);
}

function setProgress(frac, text, meta) {
    $("progFill").style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
    if (text !== undefined) $("progText").textContent = text;
    if (meta !== undefined) $("progMeta").textContent = meta;
}

function setBusy(busy) {
    state.running = busy;
    $("run").disabled = busy;
    $("run").textContent = busy ? "Generating…" : "Generate";
    $("stop").hidden = !busy;
    $("status").className = `dot ${busy ? "busy" : "on"}`;
}

async function cancel() {
    if (!state.running || !state.job) return;
    $("stop").disabled = true;
    try {
        await api(`/api/cancel?id=${state.job}`, { method: "POST" });
        log("cancel requested, stopping after the current step");
    } catch (e) {
        log(`cancel failed: ${e.message}`);
    } finally {
        $("stop").disabled = false;
    }
}

async function api(path, opts) {
    const res = await fetch(path, opts);
    const text = await res.text();
    let data;
    try {
        data = JSON.parse(text);
    } catch {
        throw new Error(text.slice(0, 200) || `HTTP ${res.status}`);
    }
    if (!res.ok && data && data.error) throw new Error(data.error);
    return data;
}

function payload() {
    return {
        prompt: $("prompt").value.trim(),
        negative: $("negative").value,
        model: $("model").value.trim(),
        width: Number($("width").value),
        height: Number($("height").value),
        steps: Number($("steps").value),
        cfg: Number($("cfg").value),
        seed: Number($("seed").value),
        sampler: $("sampler").value,
        spacing: "trailing",
        vpred: $("vpred").checked,
        zsnr: $("zsnr").checked,
        dynthresh: $("dynthresh").checked,
        rescale: $("rescale").checked ? 0.7 : 0,
    };
}

async function probe() {
    const dir = $("model").value.trim();
    if (!dir) {
        $("modelInfo").textContent = "Enter a folder path first.";
        return;
    }
    $("modelInfo").textContent = "scanning…";
    try {
        const d = await api(`/api/models?dir=${encodeURIComponent(dir)}`);
        $("modelInfo").innerHTML = [
            `unet: ${d.name || "-"}`,
            `vae:  ${d.vae.split(/[\\/]/).pop()}`,
        ].join("<br>");
        if (d.warnings && d.warnings.length) log(`scan: ${d.warnings.length} note(s), using ${d.name}`);
        else log(`scan: ${d.name}`);
    } catch (e) {
        $("modelInfo").textContent = e.message;
    }
}

function showImage(path, width, height, seed) {
    const url = `/api/image?path=${encodeURIComponent(path.split(/[\\/]/).pop())}&t=${Date.now()}`;
    const img = $("preview");
    img.onload = () => {
        $("canvas").classList.add("checker");
        document.querySelector(".empty").style.display = "none";
        img.hidden = false;
    };
    img.src = url;
    log(`saved ${path} (${width}x${height}, seed ${seed})`);
}

async function generate() {
    if (state.running) return;
    const body = payload();
    if (!body.prompt) {
        log("error: prompt is empty");
        $("prompt").focus();
        return;
    }
    setBusy(true);
    setProgress(0, "queued", "");
    log(`generate ${body.width}x${body.height} steps=${body.steps} cfg=${body.cfg} seed=${body.seed}`);
    try {
        const job = await api("/api/generate?async=1", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        state.job = job.id;
        poll(job.id);
    } catch (e) {
        log(`error: ${e.message}`);
        setBusy(false);
        setProgress(0, "failed", "");
    }
}

function poll(id) {
    clearTimeout(state.timer);
    state.timer = setTimeout(async () => {
        try {
            const j = await api(`/api/job?id=${id}`);
            setProgress(j.progress, j.text || "working", j.steps ? `step ${j.step}/${j.steps}` : "");
            if (j.done) {
                clearTimeout(state.timer);
                setBusy(false);
                if (j.error) {
                    log(`error: ${j.error}`);
                    setProgress(1, j.error === "cancelled" ? "cancelled" : "failed", "");
                    return;
                }
                setProgress(1, "done", `${(j.totalMs / 1000).toFixed(1)}s · ${j.secPerStep.toFixed(2)} s/it`);
                showImage(j.outPath, j.width, j.height, j.seed);
                return;
            }
            poll(id);
        } catch (e) {
            clearTimeout(state.timer);
            setBusy(false);
            log(`lost job: ${e.message}`);
        }
    }, 500);
}

async function boot() {
    $("steps").addEventListener("input", () => ($("stepsOut").textContent = $("steps").value));
    $("cfg").addEventListener("input", () => ($("cfgOut").textContent = Number($("cfg").value).toFixed(1)));
    $("probe").addEventListener("click", probe);
    $("run").addEventListener("click", generate);
    $("stop").addEventListener("click", cancel);
    $("prompt").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) generate();
    });
    try {
        const i = await api("/api/info");
        $("device").textContent = i.device || "no device yet";
        $("device").title = `sdvk ${i.version} · ${i.license || "GPL-3.0-only"}`;
        $("ver").textContent = `sdvk ${i.version}`;
        $("status").className = "dot on";
        if (i.modelDir) {
            $("model").value = i.modelDir;
            probe();
        }
        log(`connected to sdvk ${i.version}`);
    } catch (e) {
        log(`server not reachable: ${e.message}`);
        $("device").textContent = "offline";
    }
}

boot();
