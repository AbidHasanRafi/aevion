// Timeline bar: scrubber with event ticks, outage bands, a zoomable view window, and the transport controls.

import { h, checkbox } from './dom.js';
import { SPEEDS } from '../core/playback.js';
import { fmtTime, themeColors } from '../core/util.js';
import { runs } from '../data/decimate.js';

const EVENT_COLORS = { handover: '#4de8ff', rlf: '#ff4d4d', 'fault.start': '#ff9f43', 'fault.end': '#5fe0a0' };

export class TimelineBar {
  constructor(ctx, root, { onCamera, onScreenshot } = {}) {
    this.ctx = ctx; this.root = root; this.view = null;
    const pb = ctx.playback;
    this.track = h('div', { class: 'tl-track' });
    this.canvas = h('canvas', { class: 'tl-canvas' });
    this.track.append(this.canvas);
    this.time = h('span', { id: 'time' }, '0.00 / 0.00 s');
    this.playBtn = h('button', { id: 'play', onclick: () => pb.toggle(), title: 'Space' }, '❚❚ Pause');
    this.speedSel = h('select', { id: 'speed', onchange: (e) => pb.setSpeed(+e.target.value) }, SPEEDS.map((s) => h('option', { value: s }, `${s}×`)));
    this.speedSel.value = String(pb.speed);
    this.liveBtn = h('button', { id: 'golive', hidden: !ctx.source.live, onclick: () => { pb.followLive = !pb.followLive; this.refresh(); }, title: 'Follow the newest data' }, '● LIVE');
    this.camBtns = ['orbit', 'chase', 'top', 'fpv'].map((m) => h('button', { 'data-cam': m, onclick: () => onCamera && onCamera(m) }, m[0].toUpperCase() + m.slice(1)));
    const row = h('div', { id: 'ctl' },
      this.playBtn, h('button', { id: 'restart', title: 'Restart (R)', onclick: () => pb.seek(pb.t0) }, '⟲'), this.speedSel,
      checkbox('loop', pb.loop, (v) => { pb.loop = v; }), this.liveBtn,
      h('div', { class: 'sep' }), ...this.camBtns, h('div', { class: 'sep' }),
      h('button', { title: 'Screenshot (S)', onclick: () => onScreenshot && onScreenshot() }, '📷'), this.time);
    root.append(this.track, row);
    this._bind(); pb.on('playing', (p) => { this.playBtn.textContent = p ? '❚❚ Pause' : '▶ Play'; }); pb.on('speed', (s) => { this.speedSel.value = String(s); });
  }

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

  setCamActive(mode) { this.camBtns.forEach((b) => b.classList.toggle('on', b.dataset.cam === mode)); }
  refresh() { this.liveBtn.classList.toggle('on', this.ctx.playback.followLive); }

  draw() {
    const { ctx } = this, s = ctx.frames.overview, pb = ctx.playback, cv = this.canvas, w = cv.clientWidth, hgt = cv.clientHeight; if (!w || !hgt) return;
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2); if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hgt);
    const [t0, t1] = this._window(), X = (t) => ((t - t0) / (t1 - t0)) * w, mid = hgt / 2;
    g.fillStyle = 'rgba(128,140,170,.25)'; g.fillRect(0, mid - 3, w, 6);
    const full = pb.t1 > pb.t0 ? [X(pb.t0), X(pb.t1)] : [0, 0]; g.fillStyle = 'rgba(92,200,255,.28)'; g.fillRect(Math.max(full[0], 0), mid - 3, Math.max(Math.min(full[1], w) - Math.max(full[0], 0), 0), 6);
    if (s.has('x:link_outage')) { const o = s.series('x:link_outage', ctx.fleet.focus); g.fillStyle = 'rgba(255,77,77,.7)'; for (const [a, b] of runs((i) => o[i] > 0.5, s.T)) g.fillRect(X(s.t[a]), mid - 3, Math.max(X(s.t[Math.min(b, s.T - 1)]) - X(s.t[a]), 2), 6); }
    for (const ev of ctx.source.events || []) { if (ev.t < t0 || ev.t > t1) continue; g.fillStyle = EVENT_COLORS[ev.type] || '#ff6b6b'; g.fillRect(X(ev.t) - 1, 1, 2, hgt - 2); }
    const x = X(pb.time); g.fillStyle = themeColors().text; g.beginPath(); g.arc(x, mid, 5, 0, Math.PI * 2); g.fill();
    this.time.textContent = `${fmtTime(pb.time - pb.t0)} / ${fmtTime(pb.duration)} s${this.view ? '  (zoomed)' : ''}`;
  }
}
