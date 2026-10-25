// Moving obstacles (birds, aircraft, vehicles): drawn at their recorded position for the current playback time.

import * as THREE from 'three';
import { Layer } from './layer.js';
import { b64ToF32 } from '../core/util.js';
import { FrameStore, Cursor } from '../data/frames.js';

export function actorGeometry(a) {
  if (a.shape === 'box') return new THREE.BoxGeometry(a.size[0], a.size[1], a.size[2]);
  if (a.shape === 'cylinder') { const g = new THREE.CylinderGeometry(a.radius, a.radius, a.height, 24); g.rotateX(Math.PI / 2); return g; }
  return new THREE.SphereGeometry(a.radius, 24, 16);
}

export class ActorsLayer extends Layer {
  constructor(ctx) {
    const d = ctx.source.layers?.actors;
    super(ctx, 'actors', { label: 'Moving obstacles', group: 'Environment', available: !!(d && d.n > 0 && d.frames > 1) });
    if (!this.available) return;
    const t = Float64Array.from(b64ToF32(d.t)), pos = b64ToF32(d.pos);
    this.cursor = new Cursor(FrameStore.fromArrays(t, { pos: { data: pos, shape: [d.frames, d.n, 3] } }, d.n));
    this.meshes = d.shapes.map((a) => {
      const color = a.color ? new THREE.Color(a.color[0], a.color[1], a.color[2]) : new THREE.Color(0xf2733a);
      const m = new THREE.Mesh(actorGeometry(a), new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.1 }));
      m.castShadow = true; this.group.add(m); return m;
    });
  }

  update(cursor) {
    this.cursor.seek(cursor.time);
    for (let i = 0; i < this.meshes.length; i++) this.meshes[i].position.set(this.cursor.get('pos', i, 0), this.cursor.get('pos', i, 1), this.cursor.get('pos', i, 2));
  }
}
