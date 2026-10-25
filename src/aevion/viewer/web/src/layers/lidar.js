// LiDAR point cloud of the nearest recorded scan, coloured by range; and the accumulated voxel occupancy map built from all scans so far.

import * as THREE from 'three';
import { Layer } from './layer.js';
import { b64ToF32 } from '../core/util.js';
import { sampleColormap } from '../data/colormaps.js';
import { OccupancyGrid } from '../data/occupancy.js';
import { lowerBound } from '../core/util.js';

function scanDirections(c) {
  if (c.directions) return c.directions;       // explicit rays (sensor extrinsics applied) override the nominal pattern
  const dirs = [], C = c.channels, A = c.samples, v0 = c.v_fov[0] * Math.PI / 180, v1 = c.v_fov[1] * Math.PI / 180;
  for (let ch = 0; ch < C; ch++) {
    const el = C > 1 ? v0 + (v1 - v0) * ch / (C - 1) : (v0 + v1) / 2;
    for (let a = 0; a < A; a++) { const az = (-c.h_fov / 2 + c.h_fov * a / A) * Math.PI / 180; dirs.push([Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)]); }
  }
  return dirs;
}

function decodeLidar(d) {
  return d && d.cfg && d.cfg.channels ? { cfg: d.cfg, t: b64ToF32(d.t), r: b64ToF32(d.ranges), frames: d.frames, dirs: scanDirections(d.cfg) } : null;
}

/** Nearest scan index to time t (the scan times are sorted). */
function nearestScan(ld, t) {
  const i = lowerBound(ld.t, t, ld.frames);
  return Math.max(0, Math.min(i - (i > 0 && Math.abs(ld.t[i - 1] - t) < Math.abs((ld.t[i] ?? Infinity) - t) ? 1 : 0), ld.frames - 1));
}

/** World-frame hit points of scan `fi` for vehicle v, written through `emit(x, y, z, range)`. */
function scanPoints(ctx, ld, fi, v, emit, store) {
  const { cfg, dirs, r } = ld, R = cfg.channels * cfg.samples, mc = cfg.mount;
  const loc = store.locate(ld.t[fi]), q = store.quat(loc, v), tmp = new THREE.Vector3(), qq = new THREE.Quaternion(q[1], q[2], q[3], q[0]);
  const px = store.get('pos', loc, v, 0), py = store.get('pos', loc, v, 1), pz = store.get('pos', loc, v, 2);
  for (let i = 0; i < R; i++) {
    const range = r[(fi * ctx.N + v) * R + i];
    if (range < 0) continue;
    const d = dirs[i]; tmp.set(d[0] * range + mc[0], d[1] * range + mc[1], d[2] * range + mc[2]).applyQuaternion(qq);
    emit(tmp.x + px, tmp.y + py, tmp.z + pz, range);
  }
}

export class LidarLayer extends Layer {
  constructor(ctx) {
    const ld = decodeLidar(ctx.source.layers?.lidar);
    super(ctx, 'lidar', { label: 'LiDAR scan', group: 'Sensors', available: !!ld });
    this.ld = ld; this.last = -1;
    if (!ld) return;
    const n = ctx.N * ld.cfg.channels * ld.cfg.samples;
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3)); this.geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.pts = new THREE.Points(this.geo, new THREE.PointsMaterial({ size: Math.max(ctx.span / 260, 0.06), vertexColors: true, sizeAttenuation: true })); this.pts.frustumCulled = false;
    this.group.add(this.pts);
  }

  update(cursor) {
    const { ctx, ld } = this; if (!ld) return;
    const fi = nearestScan(ld, cursor.time);
    if (fi === this.last) return;
    this.last = fi;
    const R = ld.cfg.channels * ld.cfg.samples, pos = this.geo.attributes.position, col = this.geo.attributes.color, mr = ld.cfg.max_range || 30, c = [0, 0, 0];
    for (let v = 0; v < ctx.N; v++) {
      let i = 0; const base = v * R;
      scanPoints(ctx, ld, fi, v, (x, y, z, range) => { pos.setXYZ(base + i, x, y, z); sampleColormap('speed', range / mr, c); col.setXYZ(base + i, c[0], c[1], c[2]); i++; }, cursor.store);
      for (; i < R; i++) pos.setXYZ(base + i, 0, 0, -1000);
    }
    pos.needsUpdate = true; col.needsUpdate = true;
  }
}

/** Voxel map of everything the LiDAR has seen up to the current time (rebuilt when playback jumps backwards). */
export class OccupancyLayer extends Layer {
  constructor(ctx) {
    const ld = decodeLidar(ctx.source.layers?.lidar);
    super(ctx, 'occupancy', { label: 'Occupancy map (from LiDAR)', group: 'Sensors', defaultOn: false, available: !!ld });
    this.ld = ld; this.next = 0; this.lastT = -Infinity; this.drawn = -1;
    if (!ld) return;
    this.voxel = Math.max(ctx.span / 120, 0.1);
    this.grid = new OccupancyGrid(this.voxel, 40000);
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(this.voxel * 0.92, this.voxel * 0.92, this.voxel * 0.92), new THREE.MeshStandardMaterial({ roughness: 0.7, transparent: true, opacity: 0.85 }), 40000);
    this.mesh.count = 0; this.mesh.frustumCulled = false; this.group.add(this.mesh); this.m = new THREE.Matrix4();
  }

  update(cursor) {
    const { ld, ctx } = this; if (!ld) return;
    if (cursor.time < this.lastT - 1e-9) { this.grid.clear(); this.next = 0; }
    this.lastT = cursor.time;
    let added = 0;
    while (this.next < ld.frames && ld.t[this.next] <= cursor.time) {
      for (let v = 0; v < ctx.N; v++) scanPoints(ctx, ld, this.next, v, (x, y, z) => { if (this.grid.add(x, y, z)) added++; }, cursor.store);
      this.next++;
    }
    if (this.grid.version === this.drawn) return;
    this.drawn = this.grid.version;
    const e = this.grid.export(1), c = [0, 0, 0], zr = ctx.frames.overview.range('pos', 2), n = Math.min(e.count, this.mesh.instanceMatrix.count);
    for (let i = 0; i < n; i++) {
      this.m.makeTranslation(e.centers[3 * i], e.centers[3 * i + 1], e.centers[3 * i + 2]); this.mesh.setMatrixAt(i, this.m);
      sampleColormap('viridis', (e.centers[3 * i + 2] - zr[0] + 1) / (zr[1] - zr[0] + 6), c); this.mesh.setColorAt(i, new THREE.Color(c[0], c[1], c[2]));
    }
    this.mesh.count = n; this.mesh.instanceMatrix.needsUpdate = true; if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}
