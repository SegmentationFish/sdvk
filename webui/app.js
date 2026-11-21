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
  initFile: $('initFile'), initRow: $('initRow'), initPreview: $('initPreview'), initName: $('initName'),
  useLastBtn: $('useLastBtn'), clearInitBtn: $('clearInitBtn'),
  denoise: $('denoise'), denoiseV: $('denoiseV'),
  maskFile: $('maskFile'), maskRow: $('maskRow'), maskPreview: $('maskPreview'), maskName: $('maskName'),
  clearMaskBtn: $('clearMaskBtn'), maskFeather: $('maskFeather'), maskInvert: $('maskInvert'),
  notice: $('notice'), noticeText: $('noticeText'), noticeClose: $('noticeClose'),
  vpred: $('vpred'), dynthresh: $('dynthresh'), zsnr: $('zsnr'),
  prompt: $('prompt'), negative: $('negative'), stage: $('stage'),
  generateBtn: $('generateBtn'), cancelBtn: $('cancelBtn'), genInfo: $('genInfo'),
  bar: $('bar'), progressWrap: $('progressWrap'), progressText: $('progressText'), progressPct: $('progressPct'),
  result: $('result'), placeholder: $('placeholder'), download: $('download'), openOut: $('openOut'),
  lastOut: $('lastOut'), jobState: $('jobState'), metaSize: $('metaSize'), metaSeed: $('metaSeed'),
  metaTime: $('metaTime'), metaSpeed: $('metaSpeed'), log: $('log'), clearLog: $('clearLog'),
  latentFile: $('latentFile'), decodeVae: $('decodeVae'), decodeBtn: $('decodeBtn'),
  decodeStatus: $('decodeStatus'), decodeResultWrap: $('decodeResultWrap'), decodeResult: $('decodeResult'),
  decodeInfo: $('decodeInfo'),
  mobar: $('mobar'), mobGenBtn: $('mobGenBtn'), setupBtn: $('setupBtn'), mobPromptBtn: $('mobPromptBtn'),
  sheet: $('sheet'), sheetBody: $('sheetBody'), sheetTitle: $('sheetTitle'),
  railLeft: document.querySelector('.rail-left'), promptbar: document.querySelector('.promptbar'),
  railRight: document.querySelector('.rail-right'), infoCard: document.querySelector('.info-card'),
  paneSetup: document.querySelector('.pane-setup'), panePrompt: document.querySelector('.pane-prompt'),
  paneInfo: document.querySelector('.pane-info'), layout: document.querySelector('.layout'),
  stagecol: document.querySelector('.stagecol'),
  topbar: document.querySelector('.topbar'), dirBtn: $('dirBtn'), dirLabel: $('dirLabel'),
  mobInfoBtn: $('mobInfoBtn'),
};

let scanData = null;
let jobId = null;
let pollTimer = null;
let lastImage = null;
let initRef = null;
let maskRef = null;

function log(msg) {
  const t = new Date().toLocaleTimeString();
  el.log.textContent += `[${t}] ${msg}\n`;
  el.log.scrollTop = el.log.scrollHeight;
}

function setPill(node, text, cls = 'muted') {
  node.textContent = text;
  node.className = 'pill ' + cls;
}

let noticeTimer = null;
function setNotice(msg, cls = 'bad') {
  el.noticeText.textContent = msg;
  el.notice.className = 'notice ' + cls;
  el.notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(clearNotice, 6000);
}
function clearNotice() {
  el.notice.hidden = true;
  el.noticeText.textContent = '';
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

function vpredFromModel(file) {
  return /v-?pred/i.test(base(file));
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
                   'samplerSel', 'spacingSel', 'loraScale', 'denoise', 'maskFeather']) {
    data[k] = el[k].value;
  }
  data.vpred = el.vpred.checked;
  data.dynthresh = el.dynthresh.checked;
  data.zsnr = el.zsnr.checked;
  data.maskInvert = el.maskInvert.checked;
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
                   'spacingSel', 'loraScale', 'denoise', 'maskFeather']) {
    if (data[k] !== undefined) el[k].value = data[k];
  }
  if (data.modelDir) el.modelDir.value = data.modelDir;
  el.vpred.checked = data.vpred === true;
  el.dynthresh.checked = !!data.dynthresh;
  el.zsnr.checked = !!data.zsnr;
  el.maskInvert.checked = !!data.maskInvert;
  el.loraScaleV.textContent = Number(el.loraScale.value).toFixed(2);
  el.denoiseV.textContent = Number(el.denoise.value).toFixed(2);
}

async function scan() {
  const dir = el.modelDir.value.trim();
  if (!dir) { setPill(el.scanState, '请输入目录', 'bad'); return; }
  setPill(el.scanState, '扫描中', 'muted');
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
    el.vpred.checked = vpredFromModel(el.unetSel.value);
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
  if (initRef) {
    b.init = initRef.path;
    b.denoise = Number(el.denoise.value);
  }
  if (initRef && maskRef) {
    b.mask = maskRef.path;
    b.maskFeather = Number(el.maskFeather.value);
    b.maskInvert = el.maskInvert.checked;
  }
  // 8 GB cards cannot fit the resident weights plus the default 900 MB scratch at 768x1024.
  if (b.width * b.height <= 768 * 1024) b.scratchMb = 512;
  return b;
}

function setRunning(running) {
  el.generateBtn.disabled = running;
  el.cancelBtn.disabled = !running;
  el.generateBtn.textContent = running ? '生成中' : '生成';
  el.mobGenBtn.disabled = running;
  el.mobGenBtn.textContent = running ? '生成中' : '生成';
  el.progressWrap.hidden = !running;
  el.stage.classList.toggle('running', running);
  if (!running) setState('idle', 'muted');
}

const STATE_TEXT = { idle: '空闲', running: '生成中', done: '完成', error: '失败', cancelled: '已取消' };

function setState(text, cls = 'muted') {
  setPill(el.jobState, STATE_TEXT[text] || text, cls);
}

function setProgress(p, text, step, steps, secPerStep) {
  const pct = Math.max(0, Math.min(100, Math.round(p * 100)));
  el.bar.style.width = pct + '%';
  el.progressPct.textContent = pct + '%';
  el.progressText.textContent = text || (step && steps ? `step ${step}/${steps}` : '处理中');
  if (step && steps) el.genInfo.textContent = `step ${step}/${steps}` + (secPerStep ? ` · ${secPerStep.toFixed(2)} s/step` : '');
}

function showImage(path, meta = {}) {
  if (!path) return;
  lastImage = path;
  el.useLastBtn.disabled = false;
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
      const oom = /OUT_OF_DEVICE_MEMORY|OUT_OF_DEVICE|out of device/i.test(j.error);
      setState(cancelled ? 'cancelled' : 'error', cancelled ? 'muted' : 'bad');
      log(cancelled ? '已取消' : ('生成失败: ' + j.error));
      if (!cancelled)
        setNotice(oom ? '显存不足：降低画布，或先关闭其他占用显卡的程序。'
                      : ('生成失败：' + j.error));
      stopPoll();
      return;
    }
    if (j.done) {
      setRunning(false);
      setState('done', 'good');
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

function fieldError(msg, node) {
  setState('idle', 'muted');
  log(msg);
  setNotice(msg);
  if (node) node.focus();
}

async function generate() {
  const d = el.modelDir.value.trim();
  if (!d) { fieldError('先设置模型目录', el.modelDir); return; }
  if (!el.prompt.value.trim() && !initRef) {
    fieldError('请先输入正向提示词（或选择源图做图生图）', el.prompt);
    return;
  }
  if (maskRef && !initRef) { fieldError('局部重绘需要先选择源图', el.initFile); return; }
  const w = Number(el.width.value), h = Number(el.height.value);
  if (!w || !h || w % 64 || h % 64) {
    fieldError('宽高必须是 64 的倍数', w % 64 ? el.width : el.height);
    return;
  }
  clearNotice();
  save();
  setRunning(true);
  setProgress(0, '排队中…', 0, 0, 0);
  setState('running', 'muted');
  log(initRef ? (maskRef ? '提交局部重绘任务…' : '提交图生图任务…') : '提交生成任务…');
  try {
    const r = await api('/api/generate?async=1', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body()),
    });
    if (!r.id) throw new Error(r.error || '提交失败');
    stopPoll();
    jobId = r.id;
    setState('running', 'muted');
    log('任务 ' + jobId);
    const tick = async () => { await poll(); if (jobId) pollTimer = setTimeout(tick, 900); };
    pollTimer = setTimeout(tick, 500);
  } catch (e) {
    setRunning(false);
    setState('error', 'bad');
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

async function uploadRef(file) {
  const data = await readFileData(file);
  const up = await api('/api/upload', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: file.name, data }),
  });
  if (!up.ok || !up.path) throw new Error(up.error || '上传失败');
  return up.path;
}

function setInit(path, label) {
  initRef = { path, label: label || base(path) };
  el.initName.textContent = initRef.label;
  el.initPreview.src = '/api/image?path=' + encodeURIComponent(path) + '&t=' + Date.now();
  el.initRow.hidden = false;
}

function clearMask() {
  maskRef = null;
  el.maskRow.hidden = true;
  el.maskPreview.removeAttribute('src');
  el.maskFile.value = '';
}

function clearInit() {
  initRef = null;
  el.initRow.hidden = true;
  el.initPreview.removeAttribute('src');
  el.initFile.value = '';
  clearMask();
}

function setMask(path, label) {
  maskRef = { path, label: label || base(path) };
  el.maskName.textContent = maskRef.label;
  el.maskPreview.src = '/api/image?path=' + encodeURIComponent(path) + '&t=' + Date.now();
  el.maskRow.hidden = false;
}

async function pickInit(file) {
  if (!file) return;
  try {
    clearMask();
    setInit(await uploadRef(file), file.name);
    log('源图: ' + file.name);
  } catch (e) {
    setNotice('源图上传失败: ' + e.message);
  }
  el.initFile.value = '';
}

async function pickMask(file) {
  if (!file) return;
  if (!initRef) {
    setNotice('先选择源图，再上传蒙版');
    el.maskFile.value = '';
    return;
  }
  try {
    setMask(await uploadRef(file), file.name);
    log('蒙版: ' + file.name);
  } catch (e) {
    setNotice('蒙版上传失败: ' + e.message);
  }
  el.maskFile.value = '';
}

function useLastAsInit() {
  if (!lastImage) return;
  clearMask();
  setInit(lastImage, base(lastImage));
  log('源图: ' + base(lastImage));
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

// ---------- mobile sheet ----------
const MQ = window.matchMedia('(max-width: 900px)');
let sheetTrigger = null;
let sheetMountedToBody = false;

function isMobile() { return MQ.matches; }

function mountSheet() {
  if (!sheetMountedToBody) {
    document.body.appendChild(el.sheet);
    sheetMountedToBody = true;
  }
}

function syncViewportVar() {
  const h = el.topbar.offsetHeight;
  document.documentElement.style.setProperty('--topbar-h', h + 'px');
}

function mountForViewport() {
  mountSheet();
  syncViewportVar();
  const panes = [el.paneSetup, el.panePrompt, el.paneInfo];
  if (isMobile()) {
    if (el.paneSetup.parentElement !== el.sheetBody) {
      el.sheetBody.append(el.panePrompt, el.paneSetup, el.paneInfo);
    }
    el.mobar.hidden = false;
    setSheetView(el.sheet.dataset.view || 'setup');
    syncPanes();
    updateDirLabel();
  } else {
    if (el.sheet.contains(el.paneSetup)) {
      closeSheet(true);
      closeDir();
      for (const p of panes) p.remove();
      el.layout.insertBefore(el.paneSetup, el.stagecol);
      el.stagecol.insertBefore(el.panePrompt, el.stagecol.firstChild);
      el.layout.appendChild(el.paneInfo);
    }
    for (const p of panes) p.hidden = false;
    el.mobar.hidden = true;
    el.sheet.hidden = true;
    el.sheet.classList.remove('open', 'closing');
    document.body.style.overflow = '';
  }
}

function closeDir() {
  el.topbar.classList.remove('dir-open');
  el.dirBtn.setAttribute('aria-expanded', 'false');
}

function updateDirLabel() {
  const v = el.modelDir.value.trim();
  el.dirLabel.textContent = v || '设置模型目录';
  el.dirLabel.title = v;
}

function toggleDir(open) {
  const next = open === undefined ? !el.topbar.classList.contains('dir-open') : open;
  el.topbar.classList.toggle('dir-open', next);
  el.dirBtn.setAttribute('aria-expanded', String(next));
  if (next) setTimeout(() => el.modelDir.focus(), 80);
}

function openSheet(view, trigger) {
  if (!isMobile()) return;
  sheetTrigger = trigger || null;
  setSheetView(view);
  el.sheet.hidden = false;
  el.sheet.classList.remove('closing');
  el.sheet.getBoundingClientRect();
  el.sheet.classList.add('open');
  document.body.style.overflow = 'hidden';
  const focusable = el.sheetBody.querySelector('select, input:not([type="file"]), textarea');
  if (focusable && view === 'prompt') setTimeout(() => focusable.focus(), 300);
  else setTimeout(() => el.sheet.querySelector('.sheet-close').focus(), 320);
}

function bindSheetDrag() {
  const panel = el.sheet.querySelector('.sheet-panel');
  let startY = 0;
  let dy = 0;
  let active = false;
  const head = el.sheet.querySelector('.sheet-head');
  const handle = el.sheet.querySelector('.sheet-handle');
  for (const grip of [head, handle]) {
    if (!grip) continue;
    grip.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      active = true;
      startY = e.clientY;
      dy = 0;
      panel.classList.add('dragging');
      grip.setPointerCapture(e.pointerId);
    });
    grip.addEventListener('pointermove', (e) => {
      if (!active) return;
      dy = Math.max(0, e.clientY - startY);
      panel.style.transform = `translateY(${dy}px)`;
    });
    const end = () => {
      if (!active) return;
      active = false;
      panel.classList.remove('dragging');
      panel.style.transform = '';
      if (dy > 92) closeSheet();
    };
    grip.addEventListener('pointerup', end);
    grip.addEventListener('pointercancel', end);
  }
}

function closeSheet(instant) {
  if (el.sheet.hidden) return;
  el.sheet.classList.remove('open');
  el.sheet.classList.add('closing');
  document.body.style.overflow = '';
  const done = () => {
    el.sheet.hidden = true;
    el.sheet.classList.remove('closing');
    const panel = el.sheet.querySelector('.sheet-panel');
    if (panel) panel.style.transform = '';
    if (document.activeElement && el.sheet.contains(document.activeElement)) {
      document.activeElement.blur();
    }
    if (sheetTrigger && document.contains(sheetTrigger)) sheetTrigger.focus({ preventScroll: true });
    sheetTrigger = null;
  };
  if (instant) done();
  else setTimeout(done, 320);
}

const VIEW_TITLE = { setup: '参数', prompt: '提示词', info: '输出' };

function setSheetView(view) {
  el.sheet.dataset.view = view;
  el.sheetTitle.textContent = VIEW_TITLE[view] || '参数';
  for (const tab of el.sheet.querySelectorAll('.sheet-tab')) {
    tab.setAttribute('aria-selected', String(tab.dataset.view === view));
    tab.tabIndex = tab.dataset.view === view ? 0 : -1;
  }
  el.sheetBody.scrollTop = 0;
  syncPanes();
}

function syncPanes() {
  const views = { setup: el.paneSetup, prompt: el.panePrompt, info: el.paneInfo };
  const current = views[el.sheet.dataset.view] || el.paneSetup;
  for (const p of Object.values(views)) p.hidden = isMobile() && p !== current;
}

function trapFocus(e) {
  if (e.key === 'Tab') {
    const nodes = [...el.sheet.querySelectorAll('button, input, select, textarea, a[href]')]
      .filter((n) => !n.disabled && n.offsetParent !== null);
    if (!nodes.length) return;
    const first = nodes[0], last = nodes[nodes.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
}

el.dirBtn.addEventListener('click', () => toggleDir());
el.mobInfoBtn.addEventListener('click', () => openSheet('info', el.mobInfoBtn));
el.setupBtn.addEventListener('click', () => openSheet('setup', el.setupBtn));
el.mobPromptBtn.addEventListener('click', () => openSheet('prompt', el.mobPromptBtn));
el.mobGenBtn.addEventListener('click', () => {
  if (el.mobGenBtn.disabled) return;
  generate();
});
for (const node of el.sheet.querySelectorAll('[data-close]')) {
  node.addEventListener('click', () => closeSheet());
}
for (const tab of el.sheet.querySelectorAll('.sheet-tab')) {
  tab.addEventListener('click', () => setSheetView(tab.dataset.view));
  tab.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const tabs = [...el.sheet.querySelectorAll('.sheet-tab')];
    const i = tabs.indexOf(tab);
    const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
    setSheetView(next.dataset.view);
    next.focus();
  });
}
document.addEventListener('keydown', (e) => {
  if (el.sheet.hidden) return;
  if (e.key === 'Escape') { closeSheet(); return; }
  trapFocus(e);
});
MQ.addEventListener('change', mountForViewport);
window.addEventListener('resize', syncViewportVar);
el.dirBtn.addEventListener('click', () => setTimeout(syncViewportVar, 0));

// wiring
el.scanBtn.addEventListener('click', scan);
el.modelDir.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { scan(); if (isMobile()) toggleDir(false); }
  if (e.key === 'Escape' && isMobile()) toggleDir(false);
});
el.modelDir.addEventListener('input', updateDirLabel);
el.generateBtn.addEventListener('click', generate);
el.cancelBtn.addEventListener('click', cancel);
el.cleanBtn.addEventListener('click', clean);
el.decodeBtn.addEventListener('click', decodeLatent);
el.diceBtn.addEventListener('click', () => { el.seed.value = Math.floor(Math.random() * 2147483647); save(); });
el.loraScale.addEventListener('input', () => { el.loraScaleV.textContent = Number(el.loraScale.value).toFixed(2); });
el.denoise.addEventListener('input', () => { el.denoiseV.textContent = Number(el.denoise.value).toFixed(2); });
el.initFile.addEventListener('change', () => pickInit(el.initFile.files && el.initFile.files[0]));
el.maskFile.addEventListener('change', () => pickMask(el.maskFile.files && el.maskFile.files[0]));
el.clearInitBtn.addEventListener('click', clearInit);
el.clearMaskBtn.addEventListener('click', clearMask);
el.useLastBtn.addEventListener('click', useLastAsInit);
el.clearLog.addEventListener('click', () => { el.log.textContent = ''; });
el.noticeClose.addEventListener('click', clearNotice);
for (const node of [el.prompt, el.negative]) {
  node.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!el.generateBtn.disabled) generate();
    }
  });
}
el.openOut.addEventListener('click', () => { if (lastImage) window.open('/api/image?path=' + encodeURIComponent(lastImage), '_blank'); });
for (const node of [el.modelDir, el.steps, el.cfg, el.seed, el.width, el.height, el.prompt, el.negative,
                    el.samplerSel, el.spacingSel, el.vpred, el.dynthresh, el.zsnr, el.loraScale,
                    el.denoise, el.maskFeather, el.maskInvert]) {
  node.addEventListener('change', save);
}
for (const node of [el.clipLSel, el.clipGSel, el.vaeSel, el.loraSel]) {
  node.addEventListener('change', save);
}
el.unetSel.addEventListener('change', () => {
  el.vpred.checked = vpredFromModel(el.unetSel.value);
  save();
});
el.vaeSel.addEventListener('change', () => {
  if (el.decodeVae.value !== el.vaeSel.value) el.decodeVae.value = el.vaeSel.value;
  save();
});

mountForViewport();
bindSheetDrag();
restore();
info();
setInterval(info, 5000);
if (!el.modelDir.value) el.modelDir.value = 'E:/ofc-data/noobai';
scan();
log('界面就绪');
