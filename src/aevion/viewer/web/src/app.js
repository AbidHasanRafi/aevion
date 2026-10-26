// Aevion viewer: application bootstrap.
//
//   window.__AEVION_BOOT__ = { mode: 'static' | 'server' | 'live', payload | payloadUrl | base | ws, theme, autoplay, manual, maxFps }
//
// `static`  the whole run is embedded (single-file replay) or fetched once from `payloadUrl`
// `server`  `aevion view`: an overview plus fine windows fetched on demand from /api
// `live`    `aevion live`: frames stream in over a WebSocket while the simulation runs
//
// Everything the app exposes for tests, notebooks and the video exporter lives on `window.__aevion`.

import * as THREE from 'three';
import { Stage } from './scene/stage.js';
import { buildGround } from './scene/terrain.js';
import { buildWorld } from './scene/world.js';
import { Fleet } from './vehicles/fleet.js';
import { DEFAULT_SPEC } from './vehicles/procedural.js';
import { buildLayers } from './layers/index.js';
import { StaticSource, ChunkedSource, LiveSource } from './data/source.js';
import { Playback } from './core/playback.js';
import { Settings, safeStorage, autoQuality } from './core/settings.js';
import { Emitter, parseHash, buildHash, clamp } from './core/util.js';
import { $, toast } from './ui/dom.js';
import { icon } from './ui/icons.js';
import { Hud } from './ui/hud.js';
import { ChartDock } from './ui/charts.js';
import { TimelineBar } from './ui/timeline.js';
import { SidePanel } from './ui/panels.js';
import { ComparePanel } from './ui/compare.js';
import { SceneEditor } from './ui/editor.js';
import { buildHelp, bindKeys } from './ui/help.js';

const CAMERAS = ['orbit', 'chase', 'top', 'fpv'];

function banner(text, kind = 'info') {
  const b = $('#banner'); if (!b) return;
  b.textContent = text; b.dataset.kind = kind; b.hidden = !text;
}

function makeSource(boot) {
  const mode = boot.mode || (boot.payload || boot.payloadUrl ? 'static' : boot.ws ? 'live' : 'server');
  if (mode === 'static') {
    if (boot.payload) return new StaticSource(boot.payload);
    const src = new Emitter();
    src.ready = fetch(boot.payloadUrl).then((r) => { if (!r.ok) throw new Error(`${boot.payloadUrl}: HTTP ${r.status}`); return r.json(); }).then((p) => new StaticSource(p));
    return src;
  }
  if (mode === 'live') {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return new LiveSource(boot.ws || `${proto}//${location.host}/ws`, globalThis.WebSocket, { reconnect: !!boot.reconnect });
  }
  return new ChunkedSource(boot.base || '');
}

/** Extent of the run: world bounds if there are any, else the bounding box of all positions. */
function runExtent(source) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], take = (a, b) => { for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], a[k]); hi[k] = Math.max(hi[k], b[k]); } };
  const wb = source.world && source.world.bounds; if (wb) take(wb.slice(0, 3), wb.slice(3, 6));
  const s = source.frames && source.frames.overview;
  if (s && s.T && s.has('pos')) {
    const d = s.ch.get('pos'), step = Math.max(1, Math.floor((s.T * s.N) / 20000));
    for (let i = 0; i < s.T * s.N; i += step) { const o = i * 3; take([d[o], d[o + 1], d[o + 2]], [d[o], d[o + 1], d[o + 2]]); }
  }
  if (!Number.isFinite(lo[0])) return { span: 40, center: [0, 0, 5] };
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 10);
  return { span, center: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, Math.max((lo[2] + hi[2]) / 2, 0)] };
}

function gpuName(renderer) {
  try { const gl = renderer.getContext(), e = gl.getExtension('WEBGL_debug_renderer_info'); return e ? String(gl.getParameter(e.UNMASKED_RENDERER_WEBGL)) : ''; } catch { return ''; }
}

export async function boot(config = globalThis.__AEVION_BOOT__ || {}) {
  const hash = parseHash(location.hash), settings = new Settings(safeStorage());
  if (config.theme) settings.values.theme = config.theme;
  if (hash.theme) settings.values.theme = hash.theme;
  document.documentElement.dataset.theme = settings.get('theme');
  banner(config.mode === 'live' ? 'Connected, waiting for the simulation to start…' : 'Loading…');

  let source;
  try { source = makeSource(config); source = await source.ready; } catch (e) { banner(`Cannot load the run: ${e.message}`, 'error'); throw e; }
  banner(source.live && (!source.frames || source.frames.overview.T === 0) ? 'Waiting for the simulation to send data…' : '');

  const meta = source.meta, N = meta.n, spec = meta.vehicle || DEFAULT_SPEC, M = meta.M || spec.rotors.length;
  const { span, center } = runExtent(source);
  const events = new Emitter();
  const env = meta.environment || {};
  if (env.utc && settings.values.timeOfDay === undefined) {              // start the sky at the simulated local solar time
    const d = new Date(env.utc), hour = (d.getUTCHours() + d.getUTCMinutes() / 60 + (env.origin ? env.origin.lon / 15 : 0) + 24) % 24;
    settings.values.timeOfDay = Math.round(hour * 4) / 4;
  }
  const stage = new Stage($('#stage'), { quality: settings.get('quality') === 'auto' ? 'medium' : settings.get('quality'), span, center, theme: settings.get('theme') });
  if (settings.get('quality') === 'auto') stage.setQuality(autoQuality({ cores: navigator.hardwareConcurrency || 4, deviceMemory: navigator.deviceMemory || 4, renderer: gpuName(stage.renderer), nVehicles: N }));
  stage.setTimeOfDay(settings.get('timeOfDay'));
  if (env.weather && stage.scene.fog) stage.scene.fog.far = Math.min(stage.scene.fog.far, Math.max(env.weather.visibility_m * 1.5, span * 1.5));   // haze: fog closes in with the visibility

  const frames = source.frames;
  const playback = new Playback(() => [frames.overview.t0, Math.max(frames.overview.t1, frames.overview.t0)], { playing: hash.play !== '0' && config.autoplay !== false, loop: config.loop !== false });
  if (source.live) { playback.live = true; playback.followLive = true; }
  if (config.speed || hash.speed) playback.setSpeed(+(hash.speed || config.speed));

  const fleet = new Fleet(stage, spec, N, M, { span });
  if (settings.get('vehicleScale') > 0) fleet.setScale(settings.get('vehicleScale'));
  const model = (meta.vehicle && meta.vehicle.model) || meta.model;
  if (model) fleet.useModel(model);

  const ctx = { stage, source, frames, meta, N, M, spec, span, center, settings, playback, fleet, events, layerData: source.layers, editorActive: false };

  // ---- ground and world
  let ground = null, worldObj = null;
  const buildScenery = () => {
    for (const o of [ground, worldObj]) if (o) { stage.W.remove(o); o.traverse((c) => { c.geometry?.dispose?.(); }); }
    const g = buildGround(source.world, { span, center, theme: stage.theme, imagery: source.world && source.world.imagery });
    ground = g.object; stage.W.add(ground); ctx.groundHeight = g.height;
    worldObj = buildWorld(source.world, { theme: stage.theme }).object; stage.W.add(worldObj);
  };
  buildScenery();

  // ---- layers and UI
  const layers = buildLayers(ctx, (m) => { console.warn(m); toast(m, 3000); });
  const hud = new Hud(ctx, $('#title'), $('#hud'));
  const charts = new ChartDock(ctx, $('#charts'));
  const cursor = frames.cursor();
  let camMode = CAMERAS.includes(hash.cam) ? hash.cam : 'orbit';
  const setCam = (m) => { camMode = m; stage.controls.enabled = m === 'orbit' && !ctx.editorActive; timeline.setCamActive(m); if (m === 'orbit') { stage.controls.update(); } };
  const timeline = new TimelineBar(ctx, $('#timeline'), { onCamera: (m) => setCam(m), onScreenshot: () => screenshot() });
  const panel = new SidePanel(ctx, $('#drawer'), layers);
  const compare = new ComparePanel(ctx, panel, layers.find((L) => L.id === 'ghosts'));
  const editor = new SceneEditor(ctx, panel);
  const help = buildHelp();
  if (meta.editor) {                       // an editor page has no run to inspect: start in the editor
    document.body.classList.add('editor-mode'); panel.body.prepend(editor.section); $('#drawer-wrap').classList.add('open'); editor.setActive(true); editor.setTool('box');
  }
  events.on('view', (v) => { charts.setView(v); timeline.view = v; });
  events.on('editor', (on) => { stage.controls.enabled = camMode === 'orbit' && !on; });
  setCam(camMode);

  // ---- settings
  const dirty = { theme: false };
  events.on('setting', (key, value) => {
    if (key === 'theme') { document.documentElement.dataset.theme = value; stage.setTheme(value); buildScenery(); }
    else if (key === 'quality') stage.setQuality(value === 'auto' ? autoQuality({ renderer: gpuName(stage.renderer), nVehicles: N }) : value);
    else if (key === 'timeOfDay') stage.setTimeOfDay(+value);
    else if (key === 'vehicleScale') fleet.setScale(+value > 0 ? +value : (fleet.scale));
    void dirty;
  });

  // ---- live data
  if (source.live) {
    source.on('frames', () => { for (const L of layers) if (L.available) L.onFrames(); });
    source.on('event', (ev) => { for (const L of layers) L.onEvent(ev); });
    source.on('end', () => { banner(''); toast('Simulation finished', 2500); hud.setSub('finished'); playback.followLive = false; timeline.refresh(); });
    source.on('close', () => { if (!source.finished) banner('Connection to the simulation closed', 'error'); });
  }

  // ---- cameras
  const tmp = { p: new THREE.Vector3(), t: new THREE.Vector3(), d: new THREE.Vector3(), q: new THREE.Quaternion(), s: new THREE.Quaternion() };
  const fpvBasis = stage.fpvCam.quaternion.clone(), orbitFollow = { last: new THREE.Vector3(), has: false };
  let chasePos = null, activeCamera = stage.camera;

  function updateCamera(dt, snap) {
    const P = fleet.poses[Math.min(fleet.focus, N - 1)], dist = Math.max(fleet.arm * fleet.scale * 7, span * 0.06);
    stage.toThree(P.p, tmp.t);
    activeCamera = stage.camera;
    if (camMode === 'orbit') {
      if (settings.get('followFocus')) {
        if (orbitFollow.has) { tmp.d.copy(tmp.t).sub(orbitFollow.last); stage.camera.position.add(tmp.d); stage.controls.target.add(tmp.d); }
        orbitFollow.last.copy(tmp.t); orbitFollow.has = true;
      } else orbitFollow.has = false;
      stage.controls.update();
    } else if (camMode === 'chase') {
      const yaw = P.yaw, want = stage.toThree([P.p.x - Math.cos(yaw) * dist, P.p.y - Math.sin(yaw) * dist, P.p.z + dist * 0.45], tmp.p).clone();
      chasePos = chasePos && !snap ? chasePos.lerp(want, 1 - Math.exp(-dt * 5)) : want;
      stage.camera.position.copy(chasePos); stage.camera.up.set(0, 1, 0); stage.camera.lookAt(tmp.t);
    } else if (camMode === 'top') {
      stage.camera.position.copy(stage.toThree([P.p.x, P.p.y, P.p.z + Math.max(span * 0.9, dist * 3)], tmp.p)); stage.camera.up.set(0, 0, -1); stage.camera.lookAt(tmp.t);
    } else {
      const f = fleet.arm * fleet.scale * 0.35;
      tmp.s.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);                  // world group rotation
      tmp.q.copy(tmp.s).multiply(P.q).multiply(fpvBasis);
      stage.fpvCam.quaternion.copy(tmp.q);
      const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(P.q).multiplyScalar(f);
      stage.fpvCam.position.copy(stage.toThree([P.p.x + fwd.x, P.p.y + fwd.y, P.p.z + fwd.z], tmp.p));
      activeCamera = stage.fpvCam;
    }
    if (camMode !== 'orbit') stage.camera.up.set(0, camMode === 'top' ? 0 : 1, camMode === 'top' ? -1 : 0);
  }

  // ---- one frame
  let lastChart = 0, lastWall = performance.now(), frameCount = 0, slow = 0, aqSteps = 0;
  const ready = () => frames.overview.T > 0;

  function draw(dt, { spin = true, chartsNow = true } = {}) {
    if (!ready()) { stage.render(); return; }
    cursor.seek(playback.time);
    fleet.update(cursor, dt, { spinning: spin && playback.playing });
    for (const L of layers) if (L.enabled && L.available) { try { L.update(cursor, dt); } catch (e) { L.enabled = false; console.warn(`layer ${L.id} disabled: ${e.message}`); toast(`Layer '${L.label}' failed and was turned off`, 3000); } }
    updateCamera(dt, dt === 0);
    stage.followSun(stage.toThree(fleet.poses[Math.min(fleet.focus, N - 1)].p));
    stage.render(activeCamera);
    hud.update(cursor);
    timeline.draw();
    if (chartsNow && settings.get('showCharts')) { charts.draw(cursor); compare.draw(); }
    if (source.ensureDetail) source.ensureDetail(playback.time);
  }

  const minDt = config.maxFps > 0 ? 1 / config.maxFps : 0;
  function loop(now) {
    requestAnimationFrame(loop);
    if (document.hidden) { lastWall = now; return; }                  // a hidden tab renders nothing
    if (minDt && (now - lastWall) / 1000 < minDt * 0.9) return;       // optional frame-rate cap (tests, battery saving)
    const dt = Math.min((now - lastWall) / 1000, 0.1); lastWall = now;
    if (!ctx.paused) {
      const was = playback.time; playback.tick(dt); void was;
      frameCount++;
      const every = slow > 20 ? 3 : 1;
      draw(dt, { chartsNow: frameCount % every === 0 || !playback.playing });
      if (!playback.playing) lastChart = frameCount;
      // adaptive quality: step down when the frame time stays high
      if (dt > 0.045) slow++; else slow = Math.max(0, slow - 1);
      if (slow > 90 && aqSteps < 2 && settings.get('quality') === 'auto') { aqSteps++; slow = 0; const q = stage.quality === 'high' ? 'medium' : 'low'; stage.setQuality(q); toast(`Slow frames: quality lowered to ${q}`, 2200); }
    }
  }

  // ---- layout
  const stageEl = $('#stage-wrap');
  const fit = () => { const w = stageEl.clientWidth, hh = stageEl.clientHeight; if (w && hh) stage.resize(w, hh); };
  new ResizeObserver(fit).observe(stageEl); fit();

  // ---- screenshot
  function screenshot() {
    draw(0, { chartsNow: false });
    stage.canvas.toBlob((b) => { const a = document.createElement('a'); a.download = `aevion-${(meta.title || 'run').replace(/\W+/g, '_')}-${playback.time.toFixed(2)}s.png`; a.href = URL.createObjectURL(b); a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 3000); toast('Screenshot saved'); }, 'image/png');
  }

  // ---- keys
  const dtRec = () => (frames.overview.T > 1 ? frames.overview.duration / (frames.overview.T - 1) : 0.01);
  const toggleLayer = (id) => { const L = layers.find((x) => x.id === id); if (L && L.available) { L.enabled = !L.enabled; toast(`${L.label}: ${L.enabled ? 'on' : 'off'}`); } };
  const drawer = $('#drawer-wrap');
  bindKeys({
    keys: {
      ' ': () => playback.toggle(),
      ArrowLeft: (e) => playback.stepFrames(e.shiftKey ? -10 : -1, dtRec()), ArrowRight: (e) => playback.stepFrames(e.shiftKey ? 10 : 1, dtRec()),
      '[': () => playback.slower(), ']': () => playback.faster(), r: () => playback.seek(playback.t0),
      1: () => setCam('orbit'), 2: () => setCam('chase'), 3: () => setCam('top'), 4: () => setCam('fpv'),
      f: () => { settings.set('followFocus', !settings.get('followFocus')); toast(`Follow: ${settings.get('followFocus') ? 'on' : 'off'}`); },
      Tab: () => { fleet.setFocus((fleet.focus + 1) % N); hud.vsel.value = fleet.focus; },
      t: () => toggleLayer('trails'), v: () => toggleLayer('vectors'), l: () => toggleLayer('lidar'), w: () => toggleLayer('wireless'), c: () => toggleLayer('coverage'),
      h: () => document.body.classList.toggle('hide-ui'), d: () => drawer.classList.toggle('open'), s: () => screenshot(),
      '?': () => { help.hidden = !help.hidden; }, Escape: () => { help.hidden = true; },
    },
  });
  const drawerBtn = $('#drawer-toggle');
  drawerBtn.replaceChildren(icon('sidebar', { size: 18 })); drawerBtn.setAttribute('aria-controls', 'drawer-wrap');
  const syncDrawer = () => { const open = drawer.classList.contains('open'); drawerBtn.setAttribute('aria-expanded', String(open)); drawerBtn.classList.toggle('on', open); };
  new MutationObserver(syncDrawer).observe(drawer, { attributes: true, attributeFilter: ['class'] }); syncDrawer();
  drawerBtn.addEventListener('click', () => drawer.classList.toggle('open'));
  if (hash.ui === '0') document.body.classList.add('hide-ui');
  if (hash.focus) { fleet.setFocus(clamp(+hash.focus, 0, N - 1)); hud.vsel.value = fleet.focus; }
  if (hash.t !== undefined) playback.seek(playback.t0 + +hash.t);
  // keep the URL a shareable deep link (only while paused, to avoid history churn)
  setInterval(() => { if (!playback.playing && !source.live) history.replaceState(null, '', buildHash({ t: (playback.time - playback.t0).toFixed(2), cam: camMode, play: 0, focus: fleet.focus || undefined, ui: document.body.classList.contains('hide-ui') ? 0 : undefined })); }, 800);

  // ---- test / export API
  const api = {
    ctx, layers, charts, timeline, panel, editor, source, stage, fleet, playback, settings,
    get camera() { return camMode; }, setCamera: setCam,
    async renderFrame(t) {
      if (t !== undefined) playback.seek(t);
      playback.setPlaying(false); ctx.paused = true;
      if (source.ensureDetail) { for (let k = 0; k < 4; k++) { const p = source.ensureDetail(playback.time); if (!p) break; await p; } }
      draw(0, { spin: false });
      return playback.time;
    },
    resume() { ctx.paused = false; lastWall = performance.now(); },
    /** Turn the orbit camera about the vertical axis through its target (used for video sweeps). */
    orbitBy(rad) { const d = stage.camera.position.clone().sub(stage.controls.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), rad); stage.camera.position.copy(stage.controls.target).add(d); stage.controls.update(); },
    setUI(on) { document.body.classList.toggle('chrome-less', !on); },
    setTheme(t) { settings.set('theme', t); layers.forEach((L) => L.onSetting('theme', t)); events.emit('setting', 'theme', t); },
    setSize(w, hh) { document.documentElement.style.setProperty('--force-w', `${w}px`); document.documentElement.style.setProperty('--force-h', `${hh}px`); stage.resize(w, hh); },
    toDataURL(type = 'image/png', q) { return stage.canvas.toDataURL(type, q); },
    screenshot,
    stats() { return { frames: stage.frames, calls: stage.renderer.info.render.calls, triangles: stage.renderer.info.render.triangles, quality: stage.quality, layers: layers.filter((L) => L.available).map((L) => [L.id, L.enabled]), T: frames.overview.T, N, time: playback.time }; },
  };
  globalThis.__aevion = api;
  banner(source.live && !ready() ? 'Waiting for the simulation to send data…' : '');
  source.on?.('frames', () => { if (ready()) banner(''); });
  playback.seek(playback.time);
  if (config.manual) { ctx.paused = true; draw(0, { spin: false }); }                                // manual: draw only on renderFrame() (screenshots, video, tests)
  requestAnimationFrame((t) => { lastWall = t; loop(t); });
  document.body.dataset.ready = '1';
  return api;
}

if (globalThis.__AEVION_BOOT__ && !globalThis.__AEVION_NO_AUTOBOOT__) {
  boot().catch((e) => { console.error(e); banner(`Viewer error: ${e.message}`, 'error'); document.body.dataset.error = e.message; });
}
