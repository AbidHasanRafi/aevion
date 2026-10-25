// The fleet: every vehicle of the run, with poses and propeller animation.
//
// Up to `detailLimit` vehicles are drawn as full models (arms, motors, spinning propellers, LEDs). Beyond that the fleet switches to
// instanced low-detail bodies so a 50-vehicle swarm (or hundreds of vehicles) stays interactive on integrated graphics.

import * as THREE from 'three';
import { PALETTE, DEFAULT_SPEC, armLength, autoScale, buildDrone } from './procedural.js';
import { loadVehicleModel, cloneVehicle } from './gltf.js';

export class Fleet {
  constructor(stage, spec, N, M, { detailLimit = 16, span = 0 } = {}) {
    this.stage = stage; this.spec = spec || DEFAULT_SPEC; this.N = N; this.M = M;
    this.arm = armLength(this.spec); this.scale = Math.min(Math.max(autoScale(this.spec), (span * 0.009) / this.arm), 60);   // visible at any scene size
    this.detailed = N <= detailLimit;
    this.group = new THREE.Group(); stage.W.add(this.group);
    this.poses = Array.from({ length: N }, () => ({ p: new THREE.Vector3(), q: new THREE.Quaternion(), yaw: 0 }));
    this.drones = []; this.focus = 0; this.template = null;
    this._q = new Float64Array(4); this._m = new THREE.Matrix4(); this._s = new THREE.Vector3(); this._tmpQ = new THREE.Quaternion();
    this.colors = PALETTE;
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.0, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
    this.ring.visible = N > 1; this.group.add(this.ring);
    this.beacons = N <= 64 ? Array.from({ length: N }, (_, v) => this._beacon(v)) : [];   // constant-size rings: a small vehicle stays findable from far away
    this.detailed ? this._buildDetailed() : this._buildInstanced();
  }

  colorOf(v) { return this.colors[v % this.colors.length]; }

  _beacon(v) {
    if (!Fleet._ringTexture) {
      const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d');
      g.strokeStyle = '#fff'; g.lineWidth = 5; g.beginPath(); g.arc(32, 32, 26, 0, Math.PI * 2); g.stroke();
      Fleet._ringTexture = new THREE.CanvasTexture(c);
    }
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: Fleet._ringTexture, color: this.colorOf(v), transparent: true, opacity: 0.85, depthTest: false, sizeAttenuation: false }));
    s.scale.set(0.045, 0.045, 1); s.renderOrder = 9; this.group.add(s);
    return s;
  }

  _buildDetailed() {
    for (let v = 0; v < this.N; v++) { const d = buildDrone(this.spec, this.colorOf(v)); this.group.add(d); this.drones.push(d); }
  }

  _buildInstanced() {
    const { N, M, arm, spec } = this, L = arm, propR = spec.radius || arm * 0.5;
    const body = new THREE.InstancedMesh(new THREE.BoxGeometry(0.5 * L, 0.32 * L, 0.13 * L), new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.2 }), N);
    const cross = new THREE.BufferGeometry(); // two thin bars through the rotor layout
    const bars = [];
    spec.rotors.forEach((r) => { const len = Math.hypot(r[0], r[1]), a = new THREE.BoxGeometry(len, 0.06 * L, 0.04 * L); a.rotateZ(Math.atan2(r[1], r[0])); a.translate(r[0] / 2, r[1] / 2, 0); bars.push(a); });
    let n = 0; for (const b of bars) n += b.attributes.position.count;
    const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), idx = []; let o = 0, base = 0;
    for (const b of bars) { pos.set(b.attributes.position.array, o * 3); nor.set(b.attributes.normal.array, o * 3); for (const i of b.index.array) idx.push(i + base); o += b.attributes.position.count; base += b.attributes.position.count; }
    cross.setAttribute('position', new THREE.BufferAttribute(pos, 3)); cross.setAttribute('normal', new THREE.BufferAttribute(nor, 3)); cross.setIndex(idx);
    const arms = new THREE.InstancedMesh(cross, new THREE.MeshStandardMaterial({ color: 0x1b2030, roughness: 0.6 }), N);
    const discs = new THREE.InstancedMesh(new THREE.CircleGeometry(propR, 14), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }), N * M);
    body.castShadow = arms.castShadow = true;
    for (let v = 0; v < N; v++) body.setColorAt(v, new THREE.Color(this.colorOf(v)));
    if (body.instanceColor) body.instanceColor.needsUpdate = true;
    for (const m of [body, arms, discs]) { m.frustumCulled = false; this.group.add(m); }
    this.inst = { body, arms, discs };
  }

  /** Replace the procedural bodies with a glTF model (detailed fleets only). Resolves to true when the model was applied. */
  async useModel(model) {
    if (!this.detailed || !model || !model.url) return false;
    try {
      this.template = await loadVehicleModel(model, this.arm);
      for (let v = 0; v < this.N; v++) { this.group.remove(this.drones[v]); const d = cloneVehicle(this.template); this.group.add(d); this.drones[v] = d; }
      return true;
    } catch (e) { console.warn('vehicle model failed to load; keeping the procedural model', e); return false; }
  }

  setScale(s) { this.scale = s; }
  setFocus(v) { this.focus = v; }

  /** Pose all vehicles at the cursor and animate propellers (`dt` = wall-clock seconds, `playing` gates the spin). */
  update(cursor, dt, { spinning = true } = {}) {
    const { N, M, stage, spec } = this, hasRotor = cursor.has('rotor');
    for (let v = 0; v < N; v++) {
      const P = this.poses[v];
      P.p.set(cursor.get('pos', v, 0), cursor.get('pos', v, 1), cursor.get('pos', v, 2));
      cursor.quat(v, this._q); P.q.set(this._q[1], this._q[2], this._q[3], this._q[0]);
      P.yaw = cursor.has('euler') ? cursor.get('euler', v, 2) : 0;
    }
    if (this.detailed) {
      for (let v = 0; v < N; v++) {
        const d = this.drones[v], P = this.poses[v];
        d.position.copy(P.p); d.quaternion.copy(P.q); d.scale.setScalar(this.scale);
        if (hasRotor && spinning) for (let m = 0; m < Math.min(M, d.userData.props.length); m++) d.userData.props[m].g.rotation.z += d.userData.props[m].spin * (cursor.get('rotor', v, m) / spec.w_max) * 38 * dt;
      }
    } else {
      const { body, arms, discs } = this.inst, ms = this._m, sc = this._s.setScalar(this.scale), q2 = this._tmpQ;
      for (let v = 0; v < N; v++) {
        const P = this.poses[v];
        ms.compose(P.p, P.q, sc); body.setMatrixAt(v, ms); arms.setMatrixAt(v, ms);
        for (let m = 0; m < M; m++) {
          const r = spec.rotors[m % spec.rotors.length], off = new THREE.Vector3(r[0], r[1], (r[2] || 0) + 0.1 * this.arm).multiplyScalar(this.scale).applyQuaternion(P.q).add(P.p);
          q2.copy(P.q); ms.compose(off, q2, sc); discs.setMatrixAt(v * M + m, ms);
        }
      }
      body.instanceMatrix.needsUpdate = arms.instanceMatrix.needsUpdate = discs.instanceMatrix.needsUpdate = true;
    }
    for (let v = 0; v < this.beacons.length; v++) this.beacons[v].position.copy(this.poses[v].p);
    const f = this.poses[Math.min(this.focus, N - 1)];
    this.ring.position.copy(f.p); this.ring.position.z = Math.max(0.01, 0.02); this.ring.scale.setScalar(Math.max(this.arm * this.scale * 1.8, 0.4));
    this.ring.visible = N > 1;
    void stage;
  }

  /** Per-vehicle body group (detailed mode only): used to attach cameras and frustums. */
  bodyOf(v) { return this.detailed ? this.drones[v].userData.g : null; }

  dispose() { this.stage.W.remove(this.group); this.group.traverse((o) => { o.geometry?.dispose?.(); }); }
}
