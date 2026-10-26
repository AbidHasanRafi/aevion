// Transport bar: play controls, a scrubber with event ticks, outage bands and a zoomable view window, and the camera selector.

import { h, iconButton, segmented, select } from './dom.js';
import { icon } from './icons.js';
import { SPEEDS } from '../core/playback.js';
import { fmtTime, themeColors } from '../core/util.js';
import { runs } from '../data/decimate.js';

const EVENT_TOKEN = { handover: 'accent', rlf: 'bad', collision: 'bad', 'fault.start': 'warn', 'fault.end': 'good' };
const CAMERAS = [['orbit', 'Orbit'], ['chase', 'Chase'], ['top', 'Top'], ['fpv', 'FPV']];

/** Filled capsule from x0 to x1 centred on y with thickness `t`. */
function capsule(g, x0, x1, y, t) {
  if (x1 - x0 < 0.5) return;
  const r = Math.min(t / 2, (x1 - x0) / 2); g.beginPath(); g.moveTo(x0 + r, y - t / 2); g.lineTo(x1 - r, y - t / 2); g.arc(x1 - r, y, r, -Math.PI / 2, Math.PI / 2); g.lineTo(x0 + r, y + t / 2); g.arc(x0 + r, y, r, Math.PI / 2, -Math.PI / 2); g.closePath(); g.fill();
}

export class TimelineBar {
  constructor(ctx, root, { onCamera, onScreenshot } = {}) {
    this.ctx = ctx; this.root = root; this.view = null;
    const pb = ctx.playback;
    this.canvas = h('canvas', { class: 'tl-canvas', role: 'slider', 'aria-label': 'Timeline', tabindex: '0' });
    this.track = h('div', { class: 'tl-track' }, this.canvas);
    this.time = h('span', { id: 'time' }, '0.00 / 0.00 s');
    this.restartBtn = iconButton('restart', 'Restart (R)', () => pb.seek(pb.t0), { id: 'restart' });
    this.playBtn = h('button', { id: 'play', type: 'button', onclick: () => pb.toggle() });
    this._setPlaying(pb.playing);
    this.speedSel = select(SPEEDS.map((s) => [s, `${s}×`]), pb.speed, (v) => pb.setSpeed(+v));
    this.speedSel.id = 'speed'; this.speedSel.title = 'Playback speed'; this.speedSel.setAttribute('aria-label', 'Playback speed');
    this.loopBtn = iconButton('loop', 'Loop', () => { pb.loop = !pb.loop; this._setLoop(); }); this._setLoop();
    this.liveBtn = h('button', { id: 'golive', type: 'button', hidden: !ctx.source.live, title: 'Follow the newest data', onclick: () => { pb.followLive = !pb.followLive; this.refresh(); } }, 'Live');
    this.cam = segmented(CAMERAS, null, (m) => onCamera && onCamera(m), 'Camera');
    this.camBtns = this.cam.btns;
    root.append(h('div', { id: 'ctl' },
      h('div', { class: 'cluster' }, this.restartBtn, this.playBtn), this.track, this.time, this.speedSel, h('div', { class: 'cluster' }, this.loopBtn, this.liveBtn),
      h('div', { class: 'sep' }), this.cam.el, iconButton('camera', 'Screenshot (S)', () => onScreenshot && onScreenshot())));
    this._bind();
    pb.on('playing', (p) => this._setPlaying(p)); pb.on('speed', (s) => { this.speedSel.value = String(s); });
  }

  _setPlaying(p) {
    this.playBtn.replaceChildren(icon(p ? 'pause' : 'play', { size: 18 }));
    const label = p ? 'Pause (Space)' : 'Play (Space)'; this.playBtn.title = label; this.playBtn.setAttribute('aria-label', label);
  }

  _setLoop() { const on = !!this.ctx.playback.loop; this.loopBtn.classList.toggle('on', on); this.loopBtn.setAttribute('aria-pressed', String(on)); }

  _window() { const s = this.ctx.frames.overview; return this.view || [s.t0, Math.max(s.t1, s.t0 + 1e-6)]; }

  _bind() {
    const pb = this.ctx.playback; let down = false;
    const seek = (e) => { const r = this.canvas.getBoundingClientRect(), [a, b] = this._window(); pb.seek(a + Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1) * (b - a)); };
    this.canvas.addEventListener('pointerdown', (e) => { down = true; this.canvas.setPointerCapture(e.pointerId); seek(e); });
    this.canvas.addEventListener('pointermove', (e) => { if (down) seek(e); });
    this.canvas.addEventListener('pointerup', () => { down = false; });
    this.canvas.addEventListener('wheel', (e) => { e.preventDefault(); this.zoom(e.deltaY > 0 ? 1.3 : 0.77, (e.clientX - this.canvas.getBoundingClientRect().left) / this.canvas.clientWidth); }, { passive: false });
  }

  zoom(factor, anchor) {
    const s = this.ctx.frames.overview, [a, b] = this._window(), c = a + (b - a) * anchor, half = Math.max((b - a) * factor, 0.05);
    const lo = Math.max(s.t0, c - half * anchor), hi = Math.min(s.t1, lo + half);
    this.setView(hi - lo >= (s.t1 - s.t0) * 0.999 ? null : [lo, hi]);
  }

  setView(v) { this.view = v; this.ctx.events.emit('view', v); }

  setCamActive(mode) { this.cam.set(mode); }
  refresh() { const on = this.ctx.playback.followLive; this.liveBtn.classList.toggle('on', on); this.liveBtn.setAttribute('aria-pressed', String(on)); }

  draw() {
    const { ctx } = this, s = ctx.frames.overview, pb = ctx.playback, cv = this.canvas, w = cv.clientWidth, hgt = cv.clientHeight; if (!w || !hgt) return;
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2); if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
    const g = cv.getContext('2d'), C = themeColors(); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hgt);
    const [t0, t1] = this._window(), X = (t) => ((t - t0) / (t1 - t0)) * w, mid = hgt / 2, pad = 8, TH = 4;
    const clampX = (x) => Math.min(Math.max(x, pad), w - pad);
    // track: empty, then the part that holds data (live runs grow), then the played part
    g.fillStyle = C.fill2; capsule(g, pad, w - pad, mid, TH);
    if (pb.t1 > pb.t0) { g.fillStyle = C.fill; capsule(g, clampX(X(pb.t0)), clampX(X(pb.t1)), mid, TH); }
    const x = clampX(X(pb.time));
    g.fillStyle = C.accent; capsule(g, clampX(X(pb.t0)), x, mid, TH);
    // link outages sit on the track
    if (s.has('x:link_outage')) { const o = s.series('x:link_outage', ctx.fleet.focus); g.fillStyle = C.bad; for (const [a, b] of runs((i) => o[i] > 0.5, s.T)) capsule(g, clampX(X(s.t[a])), Math.max(clampX(X(s.t[Math.min(b, s.T - 1)])), clampX(X(s.t[a])) + 2), mid, TH); }
    // events: short ticks above the track, coloured by kind
    for (const ev of ctx.source.events || []) { if (ev.t < t0 || ev.t > t1) continue; g.fillStyle = C[EVENT_TOKEN[ev.type] || 'warn']; capsule(g, clampX(X(ev.t)) - 1, clampX(X(ev.t)) + 1, mid - 8, 6); }
    // thumb
    g.save(); g.shadowColor = 'rgba(0,0,0,.38)'; g.shadowBlur = 4; g.shadowOffsetY = 1; g.fillStyle = '#fff'; g.beginPath(); g.arc(x, mid, 7, 0, Math.PI * 2); g.fill(); g.restore();
    g.strokeStyle = 'rgba(0,0,0,.1)'; g.lineWidth = 0.5; g.beginPath(); g.arc(x, mid, 7, 0, Math.PI * 2); g.stroke();
    this.time.textContent = `${fmtTime(pb.time - pb.t0)} / ${fmtTime(pb.duration)} s${this.view ? ' (zoomed)' : ''}`;
    cv.setAttribute('aria-valuemin', '0'); cv.setAttribute('aria-valuemax', String(Math.round(pb.duration * 100) / 100)); cv.setAttribute('aria-valuenow', String(Math.round((pb.time - pb.t0) * 100) / 100));
  }
}
