'use strict';

const $ = (id) => document.getElementById(id);
const LS = 'sdvk.studio.v1';

const el = {
  modelDir: $('modelDir'), scanBtn: $('scanBtn'), scanState: $('scanState'), device: $('device'),
  cleanBtn: $('cleanBtn'),
  unetSel: $('unetSel'), clipLSel: $('clipLSel'), clipGSel: $('clipGSel'), vaeSel: $('vaeSel'),
  loraSel: $('loraSel'), loraScale: $('loraScale'), loraScaleV: $('loraScaleV'),
  steps: $('steps'), cfg: $('cfg'), samplerSel: $('samplerSel'), spacingSel: $('spacingSel'),
  seed: $('seed'), diceBtn: $('diceBtn'), width: $('width'), height: $('height'),
  vpred: $('vpred'), dynthresh: $('dynthresh'), zsnr: $('zsnr'),
  prompt: $('prompt'), negative: $('negative'),
  generateBtn: $('generateBtn'), cancelBtn: $('cancelBtn'), genInfo: $('genInfo'),
  bar: $('bar'), progressWrap: $('progressWrap'), progressText: $('progressText'), progressPct: $('progressPct'),
  result: $('result'), placeholder: $('placeholder'), download: $('download'), openOut: $('openOut'),
  lastOut: $('lastOut'), jobState: $('jobState'), metaSize: $('metaSize'), metaSeed: $('metaSeed'),
  metaTime: $('metaTime'), metaSpeed: $('metaSpeed'), log: $('log'), clearLog: $('clearLog'),
  latentFile: $('latentFile'), decodeVae: $('decodeVae'), decodeBtn: $('decodeBtn'),
  decodeStatus: $('decodeStatus'), decodeResultWrap: $('decodeResultWrap'), decodeResult: $('decodeResult'),
  decodeInfo: $('decodeInfo'),
};

let scanData = null;
let jobId = null;
let pollTimer = null;
let lastImage = null;

function log(msg) {
  const t = new Date().toLocaleTimeString();
  el.log.textContent += `[${t}] ${msg}\n`;
  el.log.scrollTop = el.log.scrollHeight;
}

function setPill(node, text, cls = 'muted') {
  node.textContent = text;
  node.className = 'pill ' + cls;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error(text || ('HTTP ' + res.status)); }
  if (!res.ok && !data.error) throw new Error(text || ('HTTP ' + res.status));
  return data;
}

function base(p) {
  const s = String(p || '').replace(/\\/g, '/');
  return s.slice(s.lastIndexOf('/') + 1);
}

function fillSelect(node, items, placeholder, preferred) {
  node.innerHTML = '';
  if (placeholder !== null) {
    const o = document.createElement('option');
    o.value = '';
    o.textContent = placeholder;
    node.appendChild(o);
  }
  for (const p of items || []) {
    const o = document.createElement('option');
    o.value = p;
    o.textContent = base(p);
    o.title = p;
    node.appendChild(o);
  }
  if (preferred && (items || []).includes(preferred)) node.value = preferred;
  else if (placeholder === null && (items || []).length) node.value = items[0];
}

function save() {
  const data = {};
  for (const k of ['modelDir', 'steps', 'cfg', 'seed', 'width', 'height', 'prompt', 'negative',
                   'samplerSel', 'spacingSel', 'loraScale']) {
    data[k] = el[k].value;
  }
  data.vpred = el.vpred.checked;
  data.dynthresh = el.dynthresh.checked;
  data.zsnr = el.zsnr.checked;
  data.unet = el.unetSel.value;
  data.clipL = el.clipLSel.value;
  data.clipG = el.clipGSel.value;
  data.vae = el.vaeSel.value;
  data.lora = el.loraSel.value;
  localStorage.setItem(LS, JSON.stringify(data));
}

function savedSettings() {
  try { return JSON.parse(localStorage.getItem(LS) || '{}'); } catch { return {}; }
}

function pickOption(node, value) {
  if (value && Array.from(node.options).some((o) => o.value === value)) node.value = value;
}

function restore() {
  const data = savedSettings();
  for (const k of ['steps', 'cfg', 'seed', 'width', 'height', 'prompt', 'negative', 'samplerSel',
                   'spacingSel', 'loraScale']) {
    if (data[k] !== undefined) el[k].value = data[k];
  }
  if (data.modelDir) el.modelDir.value = data.modelDir;
  el.vpred.checked = data.vpred !== false;
  el.dynthresh.checked = !!data.dynthresh;
  el.zsnr.checked = !!data.zsnr;
  el.loraScaleV.textContent = Number(el.loraScale.value).toFixed(2);
}

async function scan() {
  const dir = el.modelDir.value.trim();
  if (!dir) { setPill(el.scanState, '请输入目录', 'bad'); return; }
  setPill(el.scanState, '扫描中…', 'muted');
  try {
    const d = await api('/api/scan?dir=' + encodeURIComponent(dir));
    scanData = d;
    const unets = (d.unets || []).concat(d.checkpoints || []);
    fillSelect(el.unetSel, unets, null, d.defaults?.unet);
    fillSelect(el.clipLSel, d.clipL || [], '（自动）', d.defaults?.te1);
    fillSelect(el.clipGSel, d.clipG || [], '（自动）', d.defaults?.te2);
    fillSelect(el.vaeSel, d.vaes || [], '（自动）', d.defaults?.vae);
    fillSelect(el.decodeVae, d.vaes || [], '（自动）', d.defaults?.vae);
    fillSelect(el.loraSel, d.loras || [], '（不使用）', null);
    const saved = savedSettings();
    pickOption(el.unetSel, saved.unet);
    pickOption(el.clipLSel, saved.clipL);
    pickOption(el.clipGSel, saved.clipG);
    pickOption(el.vaeSel, saved.vae);
    pickOption(el.loraSel, saved.lora);
    if (el.vaeSel.value) pickOption(el.decodeVae, el.vaeSel.value);
    setPill(el.scanState, `${unets.length} unet · ${(d.loras || []).length} lora`, 'good');
    log(`扫描 ${dir}: unet ${unets.length}, vae ${(d.vaes || []).length}, clip ${(d.clipL || []).length}/${(d.clipG || []).length}, lora ${(d.loras || []).length}`);
    if (d.error) log('分类提示: ' + d.error);
    save();
  } catch (e) {
    setPill(el.scanState, '扫描失败', 'bad');
    log('扫描失败: ' + e.message);
  }
}

function body() {
  const b = {
    model: el.modelDir.value.trim(),
    prompt: el.prompt.value,
    negative: el.negative.value,
    width: Number(el.width.value),
    height: Number(el.height.value),
    steps: Number(el.steps.value),
    cfg: Number(el.cfg.value),
    seed: Number(el.seed.value),
    sampler: el.samplerSel.value,
    spacing: el.spacingSel.value,
    vpred: el.vpred.checked,
    dynthresh: el.dynthresh.checked,
    zsnr: el.zsnr.checked,
  };
  if (el.unetSel.value) b.unet = el.unetSel.value;
  if (el.clipLSel.value) b.te1 = el.clipLSel.value;
  if (el.clipGSel.value) b.te2 = el.clipGSel.value;
  if (el.vaeSel.value) b.vae = el.vaeSel.value;
  if (el.loraSel.value) { b.lora = el.loraSel.value; b.loraScale = Number(el.loraScale.value); }
  return b;
}

function setRunning(running) {
  el.generateBtn.disabled = running;
  el.cancelBtn.disabled = !running;
  el.generateBtn.textContent = running ? '生成中…' : '生成';
  el.progressWrap.hidden = !running;
  if (!running) el.jobState.className = 'pill muted';
}

function setProgress(p, text, step, steps, secPerStep) {
  const pct = Math.max(0, Math.min(100, Math.round(p * 100)));
  el.bar.style.width = pct + '%';
  el.progressPct.textContent = pct + '%';
  el.progressText.textContent = text || (step && steps ? `step ${step}/${steps}` : '处理中…');
  if (step && steps) el.genInfo.textContent = `step ${step}/${steps}` + (secPerStep ? ` · ${secPerStep.toFixed(2)} s/step` : '');
}

function showImage(path, meta = {}) {
  if (!path) return;
  lastImage = path;
  const url = '/api/image?path=' + encodeURIComponent(path) + '&t=' + Date.now();
  el.result.src = url;
  el.result.hidden = false;
  el.placeholder.hidden = true;
  el.download.href = url;
  el.download.hidden = false;
  el.openOut.disabled = false;
  el.lastOut.textContent = path;
  if (meta.width) el.metaSize.textContent = `${meta.width}×${meta.height}`;
  if (meta.seed !== undefined) el.metaSeed.textContent = meta.seed;
  if (meta.totalMs !== undefined) el.metaTime.textContent = (meta.totalMs / 1000).toFixed(1) + ' s';
  if (meta.secPerStep !== undefined) el.metaSpeed.textContent = meta.secPerStep.toFixed(2) + ' s/step';
}

async function poll() {
  if (!jobId) return;
  try {
    const j = await api('/api/job?id=' + encodeURIComponent(jobId));
    const p = Number(j.progress || 0);
    setProgress(p, j.text, j.step, j.steps, j.secPerStep);
    if (j.error) {
      setRunning(false);
      const cancelled = /cancel/i.test(j.error);
      setPill(el.jobState, cancelled ? 'cancelled' : 'error', cancelled ? 'muted' : 'bad');
      log(cancelled ? '已取消' : ('生成失败: ' + j.error));
      stopPoll();
      return;
    }
    if (j.done) {
      setRunning(false);
      setPill(el.jobState, 'done', 'good');
      setProgress(1, '完成', j.steps, j.steps, j.secPerStep);
      if (j.outPath) showImage(j.outPath, j);
      log(`完成 ${j.outPath || ''} · ${j.totalMs ? (j.totalMs / 1000).toFixed(1) + ' s' : ''}`);
      stopPoll();
    }
  } catch (e) {
    log('轮询失败: ' + e.message);
  }
}

function stopPoll() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
  jobId = null;
}

async function generate() {
  const d = el.modelDir.value.trim();
  if (!d) { setPill(el.scanState, '先设置模型目录', 'bad'); return; }
  if (!el.prompt.value.trim()) { log('请先输入正向提示词'); el.prompt.focus(); return; }
  const w = Number(el.width.value), h = Number(el.height.value);
  if (!w || !h || w % 64 || h % 64) { log('宽高必须是 64 的倍数'); return; }
  save();
  setRunning(true);
  setProgress(0, '排队中…', 0, 0, 0);
  setPill(el.jobState, 'running', 'muted');
  log('提交生成任务…');
  try {
    const r = await api('/api/generate?async=1', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body()),
    });
    if (!r.id) throw new Error(r.error || '提交失败');
    stopPoll();
    jobId = r.id;
    setPill(el.jobState, 'job ' + jobId, 'muted');
    log('任务 ' + jobId);
    const tick = async () => { await poll(); if (jobId) pollTimer = setTimeout(tick, 900); };
    pollTimer = setTimeout(tick, 500);
  } catch (e) {
    setRunning(false);
    setPill(el.jobState, 'error', 'bad');
    log('提交失败: ' + e.message);
  }
}

async function cancel() {
  if (!jobId) return;
  try {
    await api('/api/cancel?id=' + encodeURIComponent(jobId));
    log('已请求取消，当前步结束后停止');
  } catch (e) { log('取消失败: ' + e.message); }
}

async function clean() {
  try {
    const r = await api('/api/unload', { method: 'POST' });
    log(r.unloaded ? '已卸载模型并清理显存' : '显存已清理');
    await info();
  } catch (e) { log('清理失败: ' + e.message); }
}

function readFileData(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = () => reject(fr.error || new Error('读取失败'));
    fr.readAsDataURL(file);
  });
}

async function decodeLatent() {
  const f = el.latentFile.files && el.latentFile.files[0];
  if (!f) { el.decodeStatus.textContent = '先选择一个 latent 文件'; return; }
  el.decodeBtn.disabled = true;
  el.decodeStatus.textContent = '上传 latent…';
  el.decodeResultWrap.hidden = true;
  try {
    const data = await readFileData(f);
    const up = await api('/api/upload', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: f.name, data }),
    });
    if (!up.ok) throw new Error(up.error || '上传失败');
    el.decodeStatus.textContent = 'VAE 解码中（会先卸载常驻 UNet）…';
    const r = await api('/api/decode', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latent: up.path, vae: el.decodeVae.value, model: el.modelDir.value.trim() }),
    });
    if (!r.ok) throw new Error(r.error || '解码失败');
    el.decodeStatus.textContent = `完成 · ${(r.totalMs / 1000).toFixed(1)} s · ${r.width}×${r.height}`;
    el.decodeInfo.textContent = `${r.width}×${r.height} · ${(r.totalMs / 1000).toFixed(1)} s`;
    el.decodeResult.src = '/api/image?path=' + encodeURIComponent(r.outPath) + '&t=' + Date.now();
    el.decodeResultWrap.hidden = false;
    log('latent 解码完成: ' + r.outPath);
  } catch (e) {
    el.decodeStatus.textContent = '解码失败: ' + e.message;
    log('解码失败: ' + e.message);
  } finally {
    el.decodeBtn.disabled = false;
  }
}

let deviceName = '';
async function info() {
  try {
    const d = await api('/api/info');
    if (!deviceName) {
      try {
        const list = await api('/api/devices');
        const gpu = (list || []).find((x) => x.backend === 'vulkan') || (list || [])[0];
        if (gpu) deviceName = gpu.name + (gpu.vramMb ? ` · ${(gpu.vramMb / 1024).toFixed(0)} GB` : '');
      } catch {}
    }
    el.device.textContent = (d.device || deviceName || 'GPU') + (d.loaded ? ' · loaded' : '');
    el.device.className = 'pill ' + (d.loaded ? 'good' : 'muted');
    el.device.title = d.loaded ? ('loaded: ' + (d.model || d.modelDir)) : '未加载模型（生成时自动加载）';
  } catch {
    el.device.textContent = 'offline';
    el.device.className = 'pill bad';
  }
}

// wiring
el.scanBtn.addEventListener('click', scan);
el.modelDir.addEventListener('keydown', (e) => { if (e.key === 'Enter') scan(); });
el.generateBtn.addEventListener('click', generate);
el.cancelBtn.addEventListener('click', cancel);
el.cleanBtn.addEventListener('click', clean);
el.decodeBtn.addEventListener('click', decodeLatent);
el.diceBtn.addEventListener('click', () => { el.seed.value = Math.floor(Math.random() * 2147483647); save(); });
el.loraScale.addEventListener('input', () => { el.loraScaleV.textContent = Number(el.loraScale.value).toFixed(2); });
el.clearLog.addEventListener('click', () => { el.log.textContent = ''; });
el.openOut.addEventListener('click', () => { if (lastImage) window.open('/api/image?path=' + encodeURIComponent(lastImage), '_blank'); });
for (const node of [el.modelDir, el.steps, el.cfg, el.seed, el.width, el.height, el.prompt, el.negative,
                    el.samplerSel, el.spacingSel, el.vpred, el.dynthresh, el.zsnr, el.loraScale]) {
  node.addEventListener('change', save);
}
for (const node of [el.unetSel, el.clipLSel, el.clipGSel, el.vaeSel, el.loraSel]) {
  node.addEventListener('change', save);
}
el.vaeSel.addEventListener('change', () => {
  if (el.decodeVae.value !== el.vaeSel.value) el.decodeVae.value = el.vaeSel.value;
  save();
});

restore();
info();
setInterval(info, 5000);
if (!el.modelDir.value) el.modelDir.value = 'E:/ofc-data/noobai';
scan();
log('sdvk studio ready');
