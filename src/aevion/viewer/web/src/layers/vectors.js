// Velocity, wind and thrust vectors, and the rotor wash below each rotor.

import * as THREE from 'three';
import { Layer } from './layer.js';

export class VectorsLayer extends Layer {
  constructor(ctx) {
    super(ctx, 'vectors', { label: 'Velocity / wind / thrust', group: 'Vehicle', defaultOn: false });
    const N = ctx.N;
    this.vel = []; this.wind = [];
    for (let v = 0; v < N; v++) {
      const a = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x62f5ff, 0.3, 0.18); a.visible = false; this.group.add(a); this.vel.push(a);
      const w = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xffd23f, 0.3, 0.18); w.visible = false; this.group.add(w); this.wind.push(w);
    }
    this._a = new THREE.Vector3(); this._b = new THREE.Vector3();
  }

  update(cursor) {
    const { ctx } = this, span = ctx.span, arm = ctx.fleet.arm, sc = ctx.fleet.scale;
    for (let v = 0; v < ctx.N; v++) {
      const P = ctx.fleet.poses[v];
      this._a.set(cursor.get('vel', v, 0), cursor.get('vel', v, 1), cursor.get('vel', v, 2)); const sp = this._a.length();
      const va = this.vel[v]; va.visible = sp > 0.05;
      if (va.visible) { va.position.copy(P.p); va.setDirection(this._a.normalize()); va.setLength(Math.min(sp * span * 0.04, span * 0.5) + 0.001, 0.22 * sc * arm, 0.12 * sc * arm); }
      this._b.set(cursor.get('wind', v, 0), cursor.get('wind', v, 1), cursor.get('wind', v, 2)); const ws = this._b.length();
      const wa = this.wind[v]; wa.visible = ws > 0.05;
      if (wa.visible) { wa.position.copy(P.p); wa.position.z += arm * sc * 1.4; wa.setDirection(this._b.normalize()); wa.setLength(Math.min(ws * span * 0.03, span * 0.4) + 0.001, 0.2 * sc * arm, 0.1 * sc * arm); }
    }
    // thrust arrow of detailed models
    if (ctx.fleet.detailed && cursor.has('thrust')) {
      const g = ctx.spec.mass * 9.80665;
      for (let v = 0; v < ctx.N; v++) { const t = ctx.fleet.drones[v].userData.thrust; t.visible = true; t.setLength(Math.max(cursor.get('thrust', v, 0) / g, 0.05) * arm * 2.2, 0.12 * arm, 0.08 * arm); }
    }
  }

  onToggle(on) { if (!on && this.ctx.fleet?.detailed) for (const d of this.ctx.fleet.drones) d.userData.thrust.visible = false; }
}

/** Translucent downwash cones below the rotors of the focused vehicle, scaled with rotor speed. */
export class WashLayer extends Layer {
  constructor(ctx) {
    super(ctx, 'wash', { label: 'Rotor wash', group: 'Vehicle', defaultOn: false, available: ctx.fleet.detailed && (ctx.frames.overview.has('rotor')) });
    if (!this.available) return;
    this.cones = [];
    const geo = new THREE.ConeGeometry(1, 1, 18, 1, true); geo.translate(0, -0.5, 0); geo.rotateX(Math.PI / 2); // apex at the origin, opening along -z
    for (const r of ctx.spec.rotors) {
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x8fd4ff, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
      m.position.set(r[0], r[1], (r[2] || 0) + 0.02 * ctx.fleet.arm); this.cones.push(m);
    }
    this.holder = new THREE.Group(); this.cones.forEach((c) => this.holder.add(c)); this.group.add(this.holder);
  }

  update(cursor) {
    const { ctx } = this, f = ctx.fleet, v = f.focus, P = f.poses[v], arm = f.arm, sc = f.scale;
    this.holder.position.copy(P.p); this.holder.quaternion.copy(P.q); this.holder.scale.setScalar(sc);
    for (let m = 0; m < this.cones.length; m++) {
      const w = cursor.get('rotor', v, m) / ctx.spec.w_max, h = Math.max(w, 0) * arm * 4, r = (ctx.spec.radius || arm * 0.5) * (0.7 + 0.5 * w);
      this.cones[m].scale.set(r, r, Math.max(h, 1e-3)); this.cones[m].material.opacity = 0.05 + 0.12 * w;
    }
  }
}
