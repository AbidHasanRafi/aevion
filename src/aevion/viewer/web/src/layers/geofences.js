// Geofences and no-fly zones: translucent volumes (red = keep out, green = stay inside) with outlines and name labels.

import * as THREE from 'three';
import { Layer, textSprite } from './layer.js';
import { zoneExtent } from '../data/zones.js';

export { zoneExtent };

export function zoneGeometry(zone, span) {
  const [z0, z1] = zoneExtent(zone, span), h = z1 - z0;
  switch (zone.type) {
    case 'cylinder': { const g = new THREE.CylinderGeometry(zone.radius, zone.radius, h, 64, 1, false); g.rotateX(Math.PI / 2); g.translate(zone.center[0], zone.center[1], z0 + h / 2); return g; }
    case 'sphere': { const g = new THREE.SphereGeometry(zone.radius, 48, 32); g.translate(zone.center[0], zone.center[1], zone.center[2]); return g; }
    case 'box': { const s = zone.hi.map((v, i) => v - zone.lo[i]); const g = new THREE.BoxGeometry(s[0], s[1], s[2]); g.translate(zone.lo[0] + s[0] / 2, zone.lo[1] + s[1] / 2, zone.lo[2] + s[2] / 2); return g; }
    case 'polygon': {
      const shape = new THREE.Shape(zone.vertices.map((p) => new THREE.Vector2(p[0], p[1])));
      const g = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false }); g.translate(0, 0, z0); return g;
    }
    case 'altitude': {
      const side = span * 3, parts = [];
      for (const z of [zone.z_min, zone.z_max]) if (z !== null && z !== undefined && Number.isFinite(z)) { const p = new THREE.PlaneGeometry(side, side); p.translate(0, 0, z); parts.push(p); }
      if (!parts.length) return null;
      const merged = new THREE.BufferGeometry(); const pos = [], idx = []; let off = 0;
      for (const p of parts) { const a = p.attributes.position.array; for (const v of a) pos.push(v); for (const i of p.index.array) idx.push(i + off); off += a.length / 3; }
      merged.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); merged.setIndex(idx); return merged;
    }
    default: return null;
  }
}

export class GeofencesLayer extends Layer {
  constructor(ctx) {
    const fences = ctx.source.layers?.geofences;
    super(ctx, 'geofences', { label: 'Geofences / no-fly zones', group: 'Environment', available: !!(fences && fences.length) });
    if (!this.available) return;
    for (const f of fences) {
      const geo = zoneGeometry(f.zone, ctx.span); if (!geo) continue;
      const keepOut = f.kind !== 'inclusion', color = keepOut ? 0xff3a3a : 0x39d98a;
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: keepOut ? 0.16 : 0.06, side: THREE.DoubleSide, depthWrite: false }));
      mesh.renderOrder = 3; this.group.add(mesh);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }));
      this.group.add(edges);
      geo.computeBoundingBox(); const c = geo.boundingBox.getCenter(new THREE.Vector3()), top = geo.boundingBox.max.z;
      const label = textSprite(`${keepOut ? 'no-fly' : 'keep in'}: ${f.name || f.zone.type}`, { size: Math.max(ctx.span * 0.03, 1.5), color: keepOut ? '#ffb3b3' : '#b9f6d8' });
      label.position.set(c.x, c.y, Math.min(top, Math.max(ctx.span, 30)) + 1); this.group.add(label);
    }
  }
}
