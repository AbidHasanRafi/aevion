// Compare panel: toggle ghost runs and see their distance to the main run over time.

import { h, checkbox } from './dom.js';
import { differenceSeries, divergenceTime } from '../data/derived.js';
import { hexCss, fmt, fmtTime, themeColors } from '../core/util.js';
import { paddedRange } from '../data/decimate.js';

export class ComparePanel {
  constructor(ctx, panel, ghosts) {
    this.ctx = ctx; this.ghosts = ghosts; this.diffs = [];
    if (!ghosts || !ghosts.available) return;
    const main = ctx.frames.overview, rows = [];
    this.canvas = h('canvas', { class: 'cmp-canvas' });
    this.threshold = 1.0;
    ctx.ghostRuns.forEach((run, i) => {
      const d = differenceSeries(main, run.store, 0, 0); this.diffs.push(d);
      const div = divergenceTime(d, this.threshold);
      rows.push(h('div', { class: 'cmp-row' },
        checkbox(`${run.label || run.id || 'run ' + (i + 1)}`, true, (v) => ghosts.setVisible(i, v)),
        h('span', { class: 'sw', style: { background: hexCss(ghosts.runs[i].color) } }),
        h('span', { class: 'dim' }, `max ${fmt(d.max, 2)} m · rms ${fmt(d.rms, 2)} m · diverges > ${this.threshold} m ${div === null ? 'never' : 'at ' + fmtTime(div - main.t0) + ' s'}`)));
    });
    panel.add('Compare runs', h('div', {}, rows), h('div', { class: 'cap' }, 'Distance to this run (m)'), this.canvas);
    this.canvas.addEventListener('pointerdown', (e) => { const r = this.canvas.getBoundingClientRect(); ctx.playback.seekFraction((e.clientX - r.left) / r.width); });
  }

  draw() {
    if (!this.canvas || !this.canvas.clientWidth) return;
    const { ctx } = this, cv = this.canvas, w = cv.clientWidth, hgt = cv.clientHeight, dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hgt);
    const t0 = ctx.frames.overview.t0, t1 = ctx.frames.overview.t1; let hi = 1e-6;
    for (const d of this.diffs) hi = Math.max(hi, paddedRange(d.t, d.d, t0, t1)[1]);
    this.diffs.forEach((d, i) => {
      g.strokeStyle = hexCss(this.ghosts.runs[i].color); g.lineWidth = 1.3; g.beginPath();
      for (let k = 0; k < d.d.length; k += Math.max(1, Math.floor(d.d.length / w))) { const x = ((d.t[k] - t0) / (t1 - t0)) * w, y = hgt - (d.d[k] / hi) * (hgt - 4) - 2; if (k === 0) g.moveTo(x, y); else g.lineTo(x, y); }
      g.stroke();
    });
    const x = ((ctx.playback.time - t0) / (t1 - t0)) * w; g.strokeStyle = themeColors().text; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, hgt); g.stroke();
  }
}
