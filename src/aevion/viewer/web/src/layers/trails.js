// Trails: the path flown so far, coloured by speed, altitude, link quality or battery.

import * as THREE from 'three';
import { Layer } from './layer.js';
import { sampleColormap, sinrColor } from '../data/colormaps.js';
import { seriesOf } from '../data/derived.js';

export const COLOR_BY = ['speed', 'altitude', 'sinr', 'soc', 'none'];

export class TrailsLayer extends Layer {
  constructor(ctx) {
    super(ctx, 'trails', { label: 'Trails', group: 'Paths' });
    this.lines = []; this.count = 0; this.cap = 0; this.version = -1; this.colorBy = ctx.settings.get('trailColor') || 'speed';
    this.window = ctx.settings.get('trailSeconds') || 0; // 0 = whole path so far
    this._build();
  }

  _store() { return this.ctx.frames.overview; }

  _build() {
    for (const l of this.lines) { this.group.remove(l); l.geometry.dispose(); }
    this.lines = [];
    const s = this._store(), N = this.ctx.N;
    this.cap = Math.max(s.T, 256); this.count = 0;
    for (let v = 0; v < N; v++) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.cap * 3), 3));
      g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.cap * 3), 3));
      g.setDrawRange(0, 0);
      const line = new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true })); line.frustumCulled = false;
      this.group.add(line); this.lines.push(line);
    }
    this._ranges = {};
    this._fill(0);
  }

  _norm(kind, v, s) {
    // normalisation range of the colouring quantity, computed over the whole (possibly growing) run
    const key = kind + this.version + s.T;
    if (this._ranges[key]) return this._ranges[key];
    let lo = Infinity, hi = -Infinity;
    for (let u = 0; u < this.ctx.N; u++) { const ser = this._series(kind, u, s); for (const x of ser) { if (x < lo) lo = x; if (x > hi) hi = x; } }
    if (!(hi > lo)) { lo = 0; hi = lo + 1; }
    return (this._ranges[key] = [lo, hi]);
  }

  _series(kind, v, s) {
    this._cache = this._cache || {};
    const k = `${kind}:${v}:${s.T}`;
    if (!this._cache[k]) { this._cache = {}; this._cache[k] = kind === 'sinr' ? (s.has('x:link_sinr_db') ? s.series('x:link_sinr_db', v) : seriesOf(s, 'speed', v)) : seriesOf(s, kind === 'speed' ? 'speed' : kind === 'altitude' ? 'altitude' : 'soc', v); }
    return this._cache[k];
  }

  _fill(from) {
    const s = this._store(), kind = this.colorBy, c = [0, 0, 0];
    if (s.T > this.cap) { this._build(); return; }
    for (let v = 0; v < this.ctx.N; v++) {
      const g = this.lines[v].geometry, P = g.attributes.position.array, C = g.attributes.color.array;
      const ser = kind === 'none' ? null : this._series(kind, v, s);
      const [lo, hi] = kind === 'none' ? [0, 1] : this._norm(kind, v, s);
      for (let k = from; k < s.T; k++) {
        const o = (k * this.ctx.N + v) * 3;
        P[k * 3] = s.ch.get('pos')[o]; P[k * 3 + 1] = s.ch.get('pos')[o + 1]; P[k * 3 + 2] = s.ch.get('pos')[o + 2];
        if (kind === 'none') { C[k * 3] = 0.95; C[k * 3 + 1] = 0.95; C[k * 3 + 2] = 0.95; continue; }
        const u = (ser[k] - lo) / (hi - lo);
        if (kind === 'sinr' && s.has('x:link_sinr_db')) sinrColor(ser[k], this.ctx.layerData?.wireless?.outage_db ?? -6, c); else sampleColormap(kind === 'soc' ? 'heat' : 'speed', kind === 'soc' ? 1 - u : u, c);
        C[k * 3] = c[0]; C[k * 3 + 1] = c[1]; C[k * 3 + 2] = c[2];
      }
      g.attributes.position.needsUpdate = true; g.attributes.color.needsUpdate = true;
    }
    this.count = s.T;
  }

  onFrames() { const s = this._store(); if (s.T > this.count) this._fill(this.count); }

  onSetting(key, value) {
    if (key === 'trailColor') { this.colorBy = value; this._cache = {}; this._ranges = {}; this.version++; this._fill(0); }
    else if (key === 'trailSeconds') this.window = value;
  }

  update(cursor) {
    const s = this._store(), k1 = Math.min(cursor.overviewIndex + 1, s.T);
    let k0 = 0;
    if (this.window > 0) { const tStart = cursor.time - this.window; while (k0 < k1 && s.t[k0] < tStart) k0++; }
    for (const l of this.lines) l.geometry.setDrawRange(k0, Math.max(k1 - k0, 0));
  }
}
