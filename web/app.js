// Front-end wiring: model picker, camera controls, and the inference panel.
//
// Rendering of the trained Gaussians happens here in the browser (WebGL2);
// the "Run inference" button additionally asks the local server for a genuine
// PyTorch forward pass on GPU 1 so the two can be compared side by side.

import { SplatViewer } from './splat-viewer.js';

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');

const state = { mode: 'test', loaded: null };

let viewer = null;
let scenes = [];
let busy = false;

function toast(message, ms = 2000) {
  const el = $('toast');
  el.textContent = message;
  el.classList.add('on');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove('on'), ms);
}

function setStatus(el, message, cls = '') {
  el.textContent = message;
  el.className = `status ${cls}`;
}

function segment(container, attr, onPick) {
  container.addEventListener('click', (event) => {
    const btn = event.target.closest('button');
    if (!btn || !container.contains(btn)) return;
    for (const b of container.querySelectorAll('button')) b.classList.remove('on');
    btn.classList.add('on');
    onPick(btn.dataset[attr]);
  });
}

// ------------------------------------------------------------------ viewer

try {
  viewer = new SplatViewer(canvas);
} catch (err) {
  setStatus($('loadStatus'), `WebGL2 不可用：${err.message}`, 'err');
}

// ------------------------------------------------------------------ loading

async function loadScene(id) {
  const meta = scenes.find((s) => s.id === id);
  if (!meta || !viewer || busy) return;

  busy = true;
  $('generateBtn').disabled = true;
  setStatus($('loadStatus'), `正在读取 ${meta.nGaussians.toLocaleString()} 个高斯…`);
  const t0 = performance.now();
  try {
    const res = await fetch(meta.plyUrl);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    await viewer.load(buf, meta);
    viewer.requestSort(true);

    state.loaded = id;
    const seconds = (performance.now() - t0) / 1000;
    setStatus($('loadStatus'),
      `已生成 ${meta.nGaussians.toLocaleString()} 个高斯 · ` +
      `${(buf.byteLength / 1048576).toFixed(1)} MB · ${seconds.toFixed(1)} s`, 'ok');

    $('statCount').textContent = meta.nGaussians.toLocaleString();
    $('statPsnr').textContent = meta.valPsnr ? meta.valPsnr.toFixed(2) : '—';
    setInferencePanel(meta);
  } catch (err) {
    setStatus($('loadStatus'), `加载失败：${err.message}`, 'err');
  }
  busy = false;
  $('generateBtn').disabled = false;
}

async function loadDroppedFile(file) {
  if (!viewer) return;
  busy = true;
  setStatus($('loadStatus'), `正在解析 ${file.name}…`);
  try {
    const buf = await file.arrayBuffer();
    await viewer.load(buf, {});
    viewer.requestSort(true);
    state.loaded = null;
    setStatus($('loadStatus'),
      `${file.name} · ${viewer.count.toLocaleString()} 个高斯`, 'ok');
    $('statCount').textContent = viewer.count.toLocaleString();
    $('statPsnr').textContent = '—';
  } catch (err) {
    setStatus($('loadStatus'), `解析失败：${err.message}`, 'err');
  }
  busy = false;
}

// ------------------------------------------------------------------ inference

function setInferencePanel(meta) {
  const mode = state.mode;
  const range = $('idxRange');
  range.max = mode === 'test' ? 7 : 359;
  if (Number(range.value) > Number(range.max)) range.value = 0;
  $('idxVal').textContent = range.value;

  const wrap = $('videoWrap');
  wrap.innerHTML = '';
  if (meta.turntableUrl) {
    const video = document.createElement('video');
    video.src = meta.turntableUrl;
    video.controls = true;
    video.loop = true;
    video.muted = true;
    video.preload = 'none';
    video.title = '训练后由 PyTorch 渲染的 360° 转台';
    wrap.appendChild(video);
    const cap = document.createElement('div');
    cap.className = 'note';
    cap.textContent = '训练后 PyTorch 渲染的 360° 转台视频';
    wrap.appendChild(cap);
  }
}

async function runInference() {
  const scene = state.loaded || scenes[0]?.id;
  if (!scene) return;
  const btn = $('inferBtn');
  btn.disabled = true;
  const mode = state.mode;
  const idx = Number($('idxRange').value);
  setStatus($('inferStatus'), '正在 GPU 1 上运行前向推理…');
  $('inferOut').innerHTML = '';

  try {
    const res = await fetch('/api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene, mode, idx }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    const out = $('inferOut');
    out.className = 'imgs' + (data.groundTruth ? '' : ' single');
    const panels = [['模型渲染 / rendered', data.render]];
    if (data.groundTruth) panels.push(['真值 / ground truth', data.groundTruth]);
    for (const [caption, url] of panels) {
      const fig = document.createElement('figure');
      const img = document.createElement('img');
      img.src = `${url}?t=${Date.now()}`;
      img.alt = caption;
      const cap = document.createElement('figcaption');
      cap.textContent = caption;
      fig.append(img, cap);
      out.appendChild(fig);
    }

    const bits = [`${data.milliseconds} ms`, data.device, data.note];
    if (data.psnr) bits.unshift(`PSNR ${data.psnr.toFixed(2)} dB`);
    setStatus($('inferStatus'), bits.join(' · '), 'ok');
  } catch (err) {
    setStatus($('inferStatus'), `推理失败：${err.message}`, 'err');
  }
  btn.disabled = false;
}

// ------------------------------------------------------------------ controls

segment($('bgSeg'), 'bg', (value) => {
  const rgb = value.split(',').map(Number);
  viewer?.setBackground(rgb);
});

segment($('shSeg'), 'sh', (value) => viewer?.setShDegree(Number(value)));
segment($('sortSeg'), 'sort', (value) => {
  viewer?.setSortMode(value);
  if (value !== 'none') viewer?.requestSort(true);
});

segment($('modeSeg'), 'mode', (value) => {
  state.mode = value;
  setInferencePanel(scenes.find((s) => s.id === state.loaded) || {});
});

$('scaleRange').addEventListener('input', (event) => {
  const value = Number(event.target.value);
  $('scaleVal').textContent = `${value.toFixed(2)}×`;
  viewer?.setScale(value);
});

$('idxRange').addEventListener('input', (event) => {
  $('idxVal').textContent = event.target.value;
});

$('generateBtn').addEventListener('click', () => {
  const id = $('sceneSelect').value;
  if (id) loadScene(id);
});

$('sceneSelect').addEventListener('change', (event) => {
  const meta = scenes.find((s) => s.id === event.target.value);
  if (meta) setInferencePanel(meta);
});

$('inferBtn').addEventListener('click', runInference);

// ------------------------------------------------------------------ drop zone

const stage = $('stage');
const drop = $('drop');

stage.addEventListener('dragover', (event) => {
  event.preventDefault();
  drop.classList.add('on');
});
stage.addEventListener('dragleave', (event) => {
  if (event.relatedTarget && stage.contains(event.relatedTarget)) return;
  drop.classList.remove('on');
});
stage.addEventListener('drop', (event) => {
  event.preventDefault();
  drop.classList.remove('on');
  const file = event.dataTransfer?.files?.[0];
  if (file) loadDroppedFile(file);
});

// ------------------------------------------------------------------ main loop

let lastStats = 0;

function loop() {
  if (viewer) {
    viewer.render();
    const now = performance.now();
    if (now - lastStats > 400) {
      lastStats = now;
      $('statFps').textContent = viewer.stats.fps ? viewer.stats.fps.toFixed(0) : '—';
      $('statSort').textContent = viewer.stats.sorted
        ? `${viewer.stats.sortMs.toFixed(1)} ms` : '—';
    }
  }
  requestAnimationFrame(loop);
}

async function boot() {
  try {
    const res = await fetch('/api/scenes');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    scenes = (await res.json()).scenes || [];
  } catch {
    scenes = [];
  }

  const select = $('sceneSelect');
  select.innerHTML = '';

  if (!scenes.length) {
    const opt = document.createElement('option');
    opt.textContent = '（未发现已训练模型）';
    select.appendChild(opt);
    $('generateBtn').disabled = true;
    setStatus($('loadStatus'),
      '无法连接后端。用 .venv/bin/python web/serve.py 启动，或把 .ply 拖进窗口。', 'err');
    return;
  }

  for (const scene of scenes) {
    const opt = document.createElement('option');
    opt.value = scene.id;
    const psnr = scene.valPsnr ? ` · val PSNR ${scene.valPsnr.toFixed(2)}` : '';
    opt.textContent = `${scene.id} · ${scene.nGaussians.toLocaleString()} 高斯${psnr}`;
    select.appendChild(opt);
  }

  await loadScene(scenes[0].id);
}

// Exposed for headless verification.
window.__splatforge = { get viewer() { return viewer; }, get state() { return state; }, get scenes() { return scenes; } };

loop();
boot();
