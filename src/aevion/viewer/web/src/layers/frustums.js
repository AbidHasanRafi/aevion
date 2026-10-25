// Camera field-of-view frustums of the focused vehicle (one pyramid per configured camera sensor).

import * as THREE from 'three';
import { Layer } from './layer.js';

export function frustumPoints(cam, depth) {
  const v = (cam.fov_deg ?? 90) * Math.PI / 180, aspect = cam.aspect ?? (cam.width && cam.height ? cam.width / cam.height : 4 / 3);
  const hh = Math.tan(v / 2) * depth, hw = hh * aspect;
  return [[depth, -hw, -hh], [depth, hw, -hh], [depth, hw, hh], [depth, -hw, hh]];   // body frame: x forward, y left, z up
}

export class FrustumsLayer extends Layer {
  constructor(ctx) {
    const cams = ctx.source.layers?.camera;
    super(ctx, 'frustums', { label: 'Camera frustums', group: 'Sensors', defaultOn: false, available: !!(cams && cams.length) });
    if (!this.available) return;
    this.items = cams.map((cam) => {
      const depth = Math.min(cam.far ?? 30, ctx.span * 0.35), pts = frustumPoints(cam, depth), o = [0, 0, 0];
      const seg = []; for (const p of pts) seg.push(o, p); for (let i = 0; i < 4; i++) seg.push(pts[i], pts[(i + 1) % 4]);
      const g = new THREE.BufferGeometry().setFromPoints(seg.map((p) => new THREE.Vector3(...p)));
      const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x62f5ff, transparent: true, opacity: 0.85 }));
      const fill = new THREE.Mesh(new THREE.BufferGeometry().setFromPoints([pts[0], pts[1], pts[2], pts[0], pts[2], pts[3]].map((p) => new THREE.Vector3(...p))), new THREE.MeshBasicMaterial({ color: 0x62f5ff, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
      const holder = new THREE.Group(); holder.add(lines, fill);
      const mount = new THREE.Group(); mount.position.set(...(cam.mount || [0, 0, 0])); mount.rotation.y = -(cam.pitch_deg || 0) * Math.PI / 180; mount.add(holder);
      const root = new THREE.Group(); root.add(mount); this.group.add(root);
      return root;
    });
  }

  update() {
    const f = this.ctx.fleet, P = f.poses[f.focus];
    for (const r of this.items) { r.position.copy(P.p); r.quaternion.copy(P.q); r.scale.setScalar(f.scale); }
  }
}
