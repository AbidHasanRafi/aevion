// Reference path (the commanded trajectory) and planned waypoints with numbered markers.

import * as THREE from 'three';
import { Layer, textSprite } from './layer.js';

export class ReferenceLayer extends Layer {
  constructor(ctx) {
    const s = ctx.frames.overview;
    super(ctx, 'reference', { label: 'Reference path', group: 'Paths', available: s.has('ref') });
    if (!this.available) return;
    for (let v = 0; v < ctx.N; v++) {
      const T = s.T, p = new Float32Array(T * 3);
      for (let k = 0; k < T; k++) { const o = (k * ctx.N + v) * 3, d = s.ch.get('ref'); p.set([d[o], d[o + 1], d[o + 2]], k * 3); }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3));
      const l = new THREE.Line(g, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: ctx.span / 120, gapSize: ctx.span / 180, transparent: true, opacity: 0.55 }));
      l.computeLineDistances(); l.frustumCulled = false; this.group.add(l);
    }
  }
}

export class PlannedLayer extends Layer {
  constructor(ctx) {
    const pts = ctx.source.layers?.planned?.points;
    super(ctx, 'planned', { label: 'Planned waypoints', group: 'Paths', available: !!(pts && pts.length) });
    if (!this.available) return;
    const r = Math.max(ctx.span * 0.012, 0.15), mat = new THREE.MeshStandardMaterial({ color: 0xffd23f, emissive: 0x553d00, roughness: 0.4 });
    const line = [];
    pts.forEach((p, i) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), mat); m.position.set(p[0], p[1], p[2]); m.castShadow = true; this.group.add(m);
      const pole = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p[0], p[1], 0), new THREE.Vector3(p[0], p[1], p[2])]), new THREE.LineBasicMaterial({ color: 0xffd23f, transparent: true, opacity: 0.35 }));
      this.group.add(pole);
      const label = textSprite(String(i + 1), { size: r * 3.2 }); label.position.set(p[0], p[1], p[2] + r * 3); this.group.add(label);
      line.push(new THREE.Vector3(p[0], p[1], p[2]));
    });
    this.group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(line), new THREE.LineDashedMaterial({ color: 0xffd23f, dashSize: r * 2, gapSize: r, transparent: true, opacity: 0.7 })));
    this.group.children[this.group.children.length - 1].computeLineDistances();
  }
}
