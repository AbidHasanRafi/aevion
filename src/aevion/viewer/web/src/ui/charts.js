// Chart dock: any number of time-series charts of stored or derived channels, synchronised with playback.
//
// Click or drag on a chart to scrub; hover shows the value; the shared `view` window (set by the timeline brush or the wheel) zooms every
// chart at once. Events are drawn as ticks, link outages as shaded bands.

import { h, select, iconButton } from './dom.js';
import { channelInfo, chartCandidates } from '../data/channels.js';
import { seriesOf } from '../data/derived.js';
import { decimateMinMax, paddedRange, runs } from '../data/decimate.js';
import { niceTicks, fmtTick, fmtTime, hexCss, themeColors, CANVAS_FONT } from '../core/util.js';
import { PALETTE } from '../vehicles/procedural.js';

const EVENT_TOKEN = { handover: 'accent', rlf: 'bad', collision: 'bad', 'fault.start': 'warn', 'fault.end': 'good' };

export class ChartDock {
  constructor(ctx, root) {
    this.ctx = ctx; this.root = root; this.charts = []; this.view = null; this.hover = null;
    this.bar = h('div', { class: 'chart-bar' });
    root.append(this.bar);
    this.list = h('div', { class: 'chart-list' }); root.append(this.list);
    this._picker();
    const saved = ctx.settings.get('chartChannels');
    const defaults = saved || ['speed', 'altitude', ...(ctx.frames.overview.has('err') ? ['err'] : []), 'power', ...(ctx.frames.overview.has('x:link_sinr_db') ? ['x:link_sinr_db'] : [])];
    for (const name of defaults) if (this._available(name)) this.add(name);
  }

  _available(name) { const s = this.ctx.frames.overview; return name === 'speed' || name === 'altitude' || s.has(name); }

  _picker() {
    const cands = chartCandidates(this.ctx.frames.overview);
    const sel = select([['', 'Add chart…'], ...cands.map((n) => [n, channelInfo(n).label])], '', (v) => { if (v) { this.add(v); sel.value = ''; } });
    this.bar.append(sel);
  }

  add(name) {
    if (this.charts.some((c) => c.name === name)) return;
    const info = channelInfo(name);
    const val = h('span', { class: 'val' }), canvas = h('canvas'), box = h('div', { class: 'chart' },
      h('div', { class: 'cap' }, h('span', {}, `${info.label}${info.unit ? ' (' + info.unit + ')' : ''}`), val, iconButton('xmark', `Remove ${info.label} chart`, () => this.remove(name), { class: 'small x', size: 12 })), canvas);
    this.list.append(box);
    const c = { name, info, canvas, val, box, cache: null };
    this._bind(c); this.charts.push(c); this._save();
  }

  remove(name) { const i = this.charts.findIndex((c) => c.name === name); if (i < 0) return; this.charts[i].box.remove(); this.charts.splice(i, 1); this._save(); }
  _save() { this.ctx.settings.set('chartChannels', this.charts.map((c) => c.name)); }

  _bind(c) {
    const seek = (e) => { const r = c.canvas.getBoundingClientRect(), [t0, t1] = this._window(); this.ctx.playback.seek(t0 + Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1) * (t1 - t0)); };
    let down = false;
    c.canvas.addEventListener('pointerdown', (e) => { down = true; c.canvas.setPointerCapture(e.pointerId); seek(e); });
    c.canvas.addEventListener('pointermove', (e) => { const r = c.canvas.getBoundingClientRect(); this.hover = { c, x: e.clientX - r.left }; if (down) seek(e); });
    c.canvas.addEventListener('pointerup', () => { down = false; });
    c.canvas.addEventListener('pointerleave', () => { this.hover = null; });
    c.canvas.addEventListener('wheel', (e) => { e.preventDefault(); this.zoom(e.deltaY > 0 ? 1.25 : 0.8, (e.clientX - c.canvas.getBoundingClientRect().left) / c.canvas.clientWidth); }, { passive: false });
  }

  _window() { const s = this.ctx.frames.overview; return this.view || [s.t0, Math.max(s.t1, s.t0 + 1e-6)]; }

  setView(v) { this.view = v; }
  zoom(factor, anchor = 0.5) {
    const s = this.ctx.frames.overview, [a, b] = this._window(), c = a + (b - a) * anchor, half = Math.max((b - a) * factor, 1e-3);
    const lo = Math.max(s.t0, c - half * anchor), hi = Math.min(s.t1, lo + half);
    this.view = hi - lo >= (s.t1 - s.t0) * 0.999 ? null : [lo, hi];
    this.ctx.events.emit('view', this.view);
  }

  series(c, v) {
    const s = this.ctx.frames.overview, key = `${s.T}:${s.version}`;
    c.cache = c.cache && c.cache.key === key ? c.cache : { key, per: [] };
    if (!c.cache.per[v]) { const raw = seriesOf(s, c.name, v), k = c.info.scale; if (k !== 1) for (let i = 0; i < raw.length; i++) raw[i] *= k; c.cache.per[v] = raw; }
    return c.cache.per[v];
  }

  /** Vehicles drawn in each chart: all of them for small fleets, else the focused one plus an even sample (200 lines are noise anyway). */
  _shown(N, focus) {
    if (N <= 12) return Array.from({ length: N }, (_, v) => v);
    const out = new Set([focus]); for (let k = 0; k < 7; k++) out.add(Math.floor((k * N) / 7));
    return [...out].sort((a, b) => a - b);
  }

  draw(cursor) {
    const { ctx } = this, s = ctx.frames.overview, N = ctx.N, focus = ctx.fleet.focus, [t0, t1] = this._window();
    const events = ctx.source.events || [], dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    for (const c of this.charts) {
      const cv = c.canvas, w = cv.clientWidth, hgt = cv.clientHeight; if (!w || !hgt) continue;
      if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
      const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hgt);
      const C = themeColors(), { text: txt, dim } = C;
      const rkey = `${s.T}:${s.version}:${t0}:${t1}`;
      if (!c.range || c.range.key !== rkey) {          // the y-range over all vehicles only changes with the data or the zoom window
        let a0 = Infinity, b0 = -Infinity;
        for (let v = 0; v < N; v++) { const [a, b] = paddedRange(s.t, this.series(c, v), t0, t1, s.T); a0 = Math.min(a0, a); b0 = Math.max(b0, b); }
        c.range = { key: rkey, lo: Number.isFinite(a0) ? a0 : 0, hi: Number.isFinite(b0) ? b0 : 1 };
      }
      const { lo, hi } = c.range, shown = this._shown(N, focus);
      const X = (t) => ((t - t0) / (t1 - t0)) * w, Y = (y) => hgt - ((y - lo) / (hi - lo)) * (hgt - 4) - 2;
      // outage bands
      if (s.has('x:link_outage') && c.name.startsWith('x:link')) {
        const o = s.series('x:link_outage', focus), bands = runs((i) => o[i] > 0.5, s.T);
        g.globalAlpha = 0.16; g.fillStyle = C.bad; for (const [a, b] of bands) g.fillRect(X(s.t[a]), 0, Math.max(X(s.t[Math.min(b, s.T - 1)]) - X(s.t[a]), 1), hgt); g.globalAlpha = 1;
      }
      // grid + labels
      g.strokeStyle = C.hair; g.lineWidth = 1; g.fillStyle = dim; g.font = `10px ${CANVAS_FONT}`; g.textBaseline = 'top';
      const ticks = niceTicks(lo, hi, 3), tstep = ticks.length > 1 ? ticks[1] - ticks[0] : 0;
      for (const y of ticks) { const py = Y(y); g.beginPath(); g.moveTo(0, py); g.lineTo(w, py); g.stroke(); g.fillText(fmtTick(y, tstep), 3, py + 1); }
      // series
      for (const v of shown) {
        const col = N === 1 ? C.accent : hexCss(PALETTE[v % PALETTE.length]), y = this.series(c, v), d = decimateMinMax(s.t, y, t0, t1, w, s.T);
        g.strokeStyle = col; g.globalAlpha = v === focus ? 1 : 0.4; g.lineWidth = v === focus ? 1.6 : 1; g.lineJoin = 'round'; g.beginPath();
        for (let i = 0; i < d.n; i++) { const a = Y(d.lo[i]), b = Y(d.hi[i]); if (i === 0) g.moveTo(d.x[i], a); else g.lineTo(d.x[i], a); if (b !== a) g.lineTo(d.x[i], b); }
        g.stroke();
      }
      g.globalAlpha = 1;
      // events
      for (const ev of events) { if (ev.t < t0 || ev.t > t1) continue; g.fillStyle = C[EVENT_TOKEN[ev.type] || 'warn']; g.fillRect(X(ev.t) - 0.75, hgt - 6, 1.5, 6); }
      // playhead
      const px = X(cursor.time); g.strokeStyle = txt; g.globalAlpha = 0.85; g.lineWidth = 1; g.beginPath(); g.moveTo(px, 0); g.lineTo(px, hgt); g.stroke(); g.globalAlpha = 1;
      const idx = Math.min(cursor.overviewIndex, s.T - 1);
      c.val.textContent = Number.isFinite(this.series(c, focus)[idx]) ? `${this.series(c, focus)[idx].toFixed(2)} ${c.info.unit}` : '–';
      // hover
      if (this.hover && this.hover.c === c) {
        const t = t0 + (this.hover.x / w) * (t1 - t0), i = Math.min(Math.max(s.nearest(t), 0), s.T - 1), y = this.series(c, focus)[i];
        g.strokeStyle = dim; g.beginPath(); g.moveTo(this.hover.x, 0); g.lineTo(this.hover.x, hgt); g.stroke();
        g.fillStyle = txt; g.font = `500 11px ${CANVAS_FONT}`; const label = `${fmtTime(s.t[i] - s.t0)} · ${Number.isFinite(y) ? y.toFixed(2) : '–'}`; const tw = g.measureText(label).width;
        g.fillText(label, Math.min(this.hover.x + 6, w - tw - 4), 12);
      }
    }
  }
}
