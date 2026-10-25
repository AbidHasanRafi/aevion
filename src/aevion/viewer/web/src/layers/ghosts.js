// Comparison runs shown as translucent "ghost" vehicles with their own trails, driven by the same playback time.
//
// Each entry of payload.runs carries its own pos/quat/t (and optional extras). Ghosts are placed by *time*, so runs of different length or
// sampling are compared at the same instant; a run that ends earlier holds its last pose.

import * as THREE from 'three';
import { Layer } from './layer.js';
import { storeFromPayload } from '../data/source.js';
import { buildDrone, DEFAULT_SPEC, autoScale } from '../vehicles/procedural.js';
import { Cursor } from '../data/frames.js';

export const GHOST_COLORS = [0xff4d6d, 0x3ddc97, 0xffc857, 0x9b8cff, 0x4cc9f0, 0xf78fb3];

export function ghostStores(runs) {
  return (runs || []).map((r) => ({ ...r, store: storeFromPayload({ meta: { ...r.meta, shapes: r.meta.shapes }, ...r.channels }) }));
}

export class GhostsLayer extends Layer {
  constructor(ctx) {
    const runs = ctx.source.runs || [];
    super(ctx, 'ghosts', { label: 'Compared runs', group: 'Compare', available: runs.length > 0 });
    this.runs = [];
    if (!this.available) return;
    ctx.ghostRuns = ghostStores(runs);
    ctx.ghostRuns.forEach((run, i) => {
      const color = run.color ?? GHOST_COLORS[i % GHOST_COLORS.length], spec = run.meta.vehicle || ctx.spec || DEFAULT_SPEC, drones = [], N = run.store.N;
      for (let v = 0; v < N; v++) {
        const d = buildDrone(spec, color); d.traverse((o) => { if (o.material) { const m = Array.isArray(o.material) ? o.material : [o.material]; m.forEach((x) => { x.transparent = true; x.opacity = 0.5; x.depthWrite = false; }); } });
        d.visible = true; this.group.add(d); drones.push(d);
      }
      const s = run.store, trails = [];
      for (let v = 0; v < N; v++) {
        const p = new Float32Array(s.T * 3); for (let k = 0; k < s.T; k++) { const o = (k * N + v) * 3, d = s.ch.get('pos'); p.set([d[o], d[o + 1], d[o + 2]], k * 3); }
        const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3)); g.setDrawRange(0, 0);
        const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 })); l.frustumCulled = false; this.group.add(l); trails.push(l);
      }
      this.runs.push({ run, drones, trails, cursor: new Cursor(s), scale: autoScale(spec), color, visible: true });
    });
    this._q = new Float64Array(4);
  }

  setVisible(i, on) { if (this.runs[i]) { this.runs[i].visible = on; this.runs[i].drones.forEach((d) => { d.visible = on; }); this.runs[i].trails.forEach((l) => { l.visible = on; }); } }

  update(cursor) {
    for (const r of this.runs) {
      if (!r.visible) continue;
      const c = r.cursor.seek(cursor.time + (r.run.offset || 0)), N = r.run.store.N, idx = c.store.locate(c.time).i0;
      for (let v = 0; v < N; v++) {
        const d = r.drones[v];
        d.position.set(c.get('pos', v, 0), c.get('pos', v, 1), c.get('pos', v, 2)); c.quat(v, this._q); d.quaternion.set(this._q[1], this._q[2], this._q[3], this._q[0]); d.scale.setScalar(r.scale);
        r.trails[v].geometry.setDrawRange(0, idx + 1);
      }
    }
  }
}
