// Radio-environment-map slices: SINR over a horizontal plane at one or several altitudes; "volume" mode stacks all slices.
//
// Python sends each slice as a uint8 grid (SINR quantised between vmin and vmax) so a 100 x 100 slice costs 10 kB.

import * as THREE from 'three';
import { Layer } from './layer.js';
import { sampleColormap } from '../data/colormaps.js';
import { b64ToBytes } from '../core/util.js';

export function sliceTexture(slice, outageDb) {
  const nx = slice.xs.length, ny = slice.ys.length, q = b64ToBytes(slice.u8), rgba = new Uint8Array(nx * ny * 4), c = [0, 0, 0];
  const span = slice.vmax - slice.vmin;
  for (let i = 0; i < nx * ny; i++) {
    const db = slice.vmin + (q[i] / 255) * span;
    sampleColormap('viridis', (db - slice.vmin) / (span || 1), c);
    const bad = db < outageDb;
    rgba[4 * i] = bad ? 200 : c[0] * 255; rgba[4 * i + 1] = bad ? 40 : c[1] * 255; rgba[4 * i + 2] = bad ? 40 : c[2] * 255; rgba[4 * i + 3] = 255;
  }
  const tex = new THREE.DataTexture(rgba, nx, ny, THREE.RGBAFormat); tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter; tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
  return tex;
}

export class CoverageLayer extends Layer {
  constructor(ctx) {
    const rem = ctx.source.layers?.rem;
    super(ctx, 'coverage', { label: 'SINR coverage map', group: 'Wireless', defaultOn: false, available: !!(rem && rem.slices && rem.slices.length) });
    if (!this.available) return;
    this.rem = rem; this.planes = []; this.mode = ctx.settings.get('coverageMode') || 'slice'; this.alt = ctx.settings.get('coverageAlt') ?? null;
    rem.slices.forEach((s) => {
      const nx = s.xs.length, ny = s.ys.length, w = s.xs[nx - 1] - s.xs[0], h = s.ys[ny - 1] - s.ys[0];
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w * (nx / (nx - 1)), h * (ny / (ny - 1))), new THREE.MeshBasicMaterial({ map: sliceTexture(s, rem.outage_db ?? -6), transparent: true, opacity: 0.7, side: THREE.DoubleSide, depthWrite: false }));
      m.position.set((s.xs[0] + s.xs[nx - 1]) / 2, (s.ys[0] + s.ys[ny - 1]) / 2, s.altitude); m.renderOrder = 2;
      this.group.add(m); this.planes.push(m);
    });
    this._apply();
  }

  get altitudes() { return this.rem.slices.map((s) => s.altitude); }

  /** Choose the slice nearest to `alt` (slice mode) or show them all (volume mode). */
  _apply() {
    const alts = this.altitudes;
    const a = this.alt ?? alts[0]; let best = 0; alts.forEach((x, i) => { if (Math.abs(x - a) < Math.abs(alts[best] - a)) best = i; });
    this.planes.forEach((p, i) => { p.visible = this.mode === 'volume' || i === best; p.material.opacity = this.mode === 'volume' ? 0.28 : 0.7; });
    this.current = alts[best];
  }

  onSetting(key, value) {
    if (key === 'coverageMode') this.mode = value; else if (key === 'coverageAlt') this.alt = value; else return;
    this._apply();
  }
}
