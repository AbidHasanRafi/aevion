// Antenna pattern lobes: the radiation pattern of each cell (fixed in the world) and of the UAV antenna (moving with the body).
//
// Python samples each pattern on a (azimuth, elevation) grid and sends the gains in dBi. The lobe radius is proportional to the
// field strength, 10^((G - G_max) / 20), clipped at -30 dB, and the surface is coloured by gain.

import * as THREE from 'three';
import { Layer } from './layer.js';
import { b64ToF32 } from '../core/util.js';
import { sampleColormap } from '../data/colormaps.js';

export function lobeGeometry(lobe, floorDb = -30) {
  const naz = lobe.n_az, nel = lobe.n_el, g = b64ToF32(lobe.gain_db), peak = lobe.peak_dbi ?? Math.max(...g);
  const pos = new Float32Array(naz * nel * 3), col = new Float32Array(naz * nel * 3), c = [0, 0, 0];
  for (let j = 0; j < nel; j++) for (let i = 0; i < naz; i++) {
    const az = lobe.az_deg[i] * Math.PI / 180, el = lobe.el_deg[j] * Math.PI / 180, gain = g[j * naz + i];
    const r = Math.pow(10, Math.max(gain - peak, floorDb) / 20), k = (j * naz + i) * 3;
    pos[k] = r * Math.cos(el) * Math.cos(az); pos[k + 1] = r * Math.cos(el) * Math.sin(az); pos[k + 2] = r * Math.sin(el);
    sampleColormap('heat', (Math.max(gain - peak, floorDb) - floorDb) / -floorDb, c); col[k] = c[0]; col[k + 1] = c[1]; col[k + 2] = c[2];
  }
  const idx = [];
  for (let j = 0; j < nel - 1; j++) for (let i = 0; i < naz; i++) { const i2 = (i + 1) % naz, a = j * naz + i, b = j * naz + i2, d = (j + 1) * naz + i, e = (j + 1) * naz + i2; idx.push(a, b, d, b, e, d); }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(col, 3)); geo.setIndex(idx); geo.computeVertexNormals();
  return geo;
}

export class BeamsLayer extends Layer {
  constructor(ctx) {
    const b = ctx.source.layers?.beams;
    super(ctx, 'beams', { label: 'Antenna patterns', group: 'Wireless', defaultOn: false, available: !!(b && b.lobes && b.lobes.length) });
    if (!this.available) return;
    this.scale = Math.max(ctx.span * 0.1, 3);
    this.cellLobes = []; this.uavLobe = null;
    const mat = () => new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, opacity: 0.55, roughness: 0.6, side: THREE.DoubleSide, depthWrite: false });
    for (const lobe of b.lobes) {
      const mesh = new THREE.Mesh(lobeGeometry(lobe), mat());
      if (lobe.kind === 'cell') {
        const cell = (ctx.source.layers.wireless?.cells || [])[lobe.cell];
        if (!cell) continue;
        mesh.position.set(cell.pos[0], cell.pos[1], cell.pos[2]); mesh.rotation.z = (cell.yaw_deg ?? 0) * Math.PI / 180; mesh.scale.setScalar(this.scale * 1.6);
        this.group.add(mesh); this.cellLobes.push(mesh);
      } else if (lobe.kind === 'uav') { this.uavLobe = mesh; mesh.scale.setScalar(this.scale * 0.5); this.group.add(mesh); }
    }
  }

  update() {
    if (!this.uavLobe) return;
    const f = this.ctx.fleet, P = f.poses[f.focus];
    this.uavLobe.position.copy(P.p); this.uavLobe.quaternion.copy(P.q);
  }
}
