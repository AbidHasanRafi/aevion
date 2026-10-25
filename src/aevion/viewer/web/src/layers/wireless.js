// Wireless layer: cells (mast, sector wedge, label), links coloured by SINR, outage halo, handover and failure pulses.
//
// Link channels come from the frame store: x:link_sinr_db, x:link_serving, x:link_los, x:link_outage. Handover / RLF / outage pulses are drawn
// from the run's events (a pure function of time, so scrubbing works), not from per-frame state.

import * as THREE from 'three';
import { Layer, textSprite } from './layer.js';
import { sinrColor } from '../data/colormaps.js';

const PULSE = 1.4; // seconds of simulated time that a pulse is visible

export class WirelessLayer extends Layer {
  constructor(ctx) {
    const w = ctx.source.layers?.wireless;
    super(ctx, 'wireless', { label: 'Radio links and cells', group: 'Wireless', available: !!w && ctx.frames.overview.has('x:link_sinr_db') });
    this.w = w;
    if (!this.available) return;
    this.mesh = w.kind === 'mesh';
    this.outageDb = w.outage_db ?? -6;
    this.cells = (w.cells || []).map((c) => ({ ...c }));
    this.solid = []; this.dashed = []; this.halos = [];
    const r = Math.max(ctx.span * 0.012, 0.2);
    const mastMat = new THREE.MeshStandardMaterial({ color: 0xdddddd }), headMat = new THREE.MeshStandardMaterial({ color: 0xff3a5c, emissive: 0x550011 });
    this.cells.forEach((cell, i) => {
      const g = new THREE.Group(), p = cell.pos, h = Math.max(p[2], 1);
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.25, r * 0.4, h, 8), mastMat); mast.rotation.x = Math.PI / 2; mast.position.set(p[0], p[1], h / 2); mast.castShadow = true; g.add(mast);
      const head = new THREE.Mesh(new THREE.ConeGeometry(r * 2.2, r * 3.2, 3), headMat); head.rotation.x = Math.PI / 2; head.position.set(p[0], p[1], p[2] + r * 1.6); g.add(head);
      const wedge = this._wedge(cell, ctx.span); if (wedge) g.add(wedge);
      const label = textSprite(`cell ${i}`, { size: r * 3 }); label.position.set(p[0], p[1], p[2] + r * 6); g.add(label);
      this.group.add(g);
    });
    for (let v = 0; v < ctx.N; v++) {
      const geo = () => new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const s = new THREE.Line(geo(), new THREE.LineBasicMaterial({ color: 0x66ff99, transparent: true, opacity: 0.9 })); s.frustumCulled = false;
      const d = new THREE.Line(geo(), new THREE.LineDashedMaterial({ color: 0x66ff99, dashSize: r * 2, gapSize: r * 1.4, transparent: true, opacity: 0.9 })); d.frustumCulled = false;
      const halo = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0xff3a3a, transparent: true, opacity: 0.22, depthWrite: false })); halo.visible = false;
      this.group.add(s, d, halo); this.solid.push(s); this.dashed.push(d); this.halos.push(halo);
    }
    this.pulses = new THREE.Group(); this.group.add(this.pulses); this.pulseMeshes = [];
    for (let i = 0; i < 24; i++) { const m = new THREE.Mesh(new THREE.RingGeometry(0.85, 1, 40), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, side: THREE.DoubleSide, depthWrite: false })); m.visible = false; this.pulses.add(m); this.pulseMeshes.push(m); }
    this._c = [0, 0, 0]; this._a = new THREE.Vector3(); this._b = new THREE.Vector3();
  }

  _wedge(cell, span) {
    const bw = (cell.beamwidth_deg ?? 65) * Math.PI / 180, R = Math.max(span * 0.18, 5), seg = 24, yaw = (cell.yaw_deg ?? 0) * Math.PI / 180;
    if (cell.omni) return null;
    const verts = [cell.pos[0], cell.pos[1], 0.05];
    for (let i = 0; i <= seg; i++) { const a = yaw - bw + (2 * bw * i) / seg; verts.push(cell.pos[0] + R * Math.cos(a), cell.pos[1] + R * Math.sin(a), 0.05); }
    const idx = []; for (let i = 1; i <= seg; i++) idx.push(0, i, i + 1);
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3)); g.setIndex(idx);
    return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xff3a5c, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
  }

  update(cursor) {
    const { ctx } = this, N = ctx.N, f = ctx.fleet, c = this._c, sc = f.scale * f.arm;
    for (let v = 0; v < N; v++) {
      const P = f.poses[v], sinr = cursor.get('x:link_sinr_db', v, 0), serving = Math.round(cursor.get('x:link_serving', v, 0));
      const los = !cursor.has('x:link_los') || cursor.get('x:link_los', v, 0) > 0.5, out = cursor.has('x:link_outage') ? cursor.get('x:link_outage', v, 0) > 0.5 : sinr < this.outageDb;
      if (this.mesh) { const q = f.poses[Math.min(Math.max(serving, 0), N - 1)]; this._b.copy(q.p); }
      else { const cell = this.cells[Math.min(Math.max(serving, 0), this.cells.length - 1)]; this._b.set(cell.pos[0], cell.pos[1], cell.pos[2] + Math.max(ctx.span * 0.012, 0.2) * 1.6); }
      sinrColor(sinr, this.outageDb, c);
      const L = los ? this.solid[v] : this.dashed[v], O = los ? this.dashed[v] : this.solid[v];
      O.visible = false; L.visible = !(this.mesh && serving === v);
      const pos = L.geometry.attributes.position; pos.setXYZ(0, P.p.x, P.p.y, P.p.z); pos.setXYZ(1, this._b.x, this._b.y, this._b.z); pos.needsUpdate = true;
      L.geometry.computeBoundingSphere(); if (!los) L.computeLineDistances();
      L.material.color.setRGB(c[0], c[1], c[2]); L.material.opacity = out ? 0.45 : 0.95;
      const h = this.halos[v]; h.visible = out; if (out) { h.position.copy(P.p); h.scale.setScalar(Math.max(sc * 1.6, ctx.span * 0.02)); }
    }
    this._pulses(cursor.time);
  }

  /** Rings at the vehicle for handovers (cyan) and radio-link failures (red) in the last PULSE seconds of simulated time. */
  _pulses(t) {
    const f = this.ctx.fleet; let n = 0;
    for (const ev of this.ctx.source.events || []) {
      if ((ev.type !== 'handover' && ev.type !== 'rlf') || ev.t > t || t - ev.t > PULSE || n >= this.pulseMeshes.length) continue;
      const m = this.pulseMeshes[n++], u = (t - ev.t) / PULSE, P = f.poses[Math.min(ev.vehicle ?? 0, this.ctx.N - 1)];
      m.visible = true; m.position.copy(P.p); m.scale.setScalar(Math.max(f.arm * f.scale * 2, this.ctx.span * 0.02) * (1 + 5 * u)); m.material.opacity = 0.9 * (1 - u);
      m.material.color.set(ev.type === 'rlf' ? 0xff3a3a : 0x4de8ff);
      m.lookAt(this.ctx.stage.camera.position); // billboard towards the active camera
    }
    for (; n < this.pulseMeshes.length; n++) this.pulseMeshes[n].visible = false;
  }
}
