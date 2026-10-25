// The 3-D stage: renderer, cameras, controls, sky, lights and the ENU world group.
//
// Archive data is in a world frame with x east, y north, z up. three.js is y-up, so every object lives in `stage.W`, a group rotated by
// -90 degrees about x: positions from the archive are used verbatim inside it.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { makeSky, sunDirection, applyTimeOfDay } from './sky.js';

export const QUALITY = {
  low: { pixelRatio: 1, shadows: false, shadowMap: 1024, antialias: false },
  medium: { pixelRatio: 1.5, shadows: true, shadowMap: 1024, antialias: true },
  high: { pixelRatio: 2, shadows: true, shadowMap: 2048, antialias: true },
};

export class Stage {
  constructor(canvas, { quality = 'high', span = 20, center = [0, 0, 0], theme = 'dark' } = {}) {
    this.canvas = canvas; this.span = span; this.center = center; this.quality = quality; this.theme = theme;
    const q = QUALITY[quality] || QUALITY.high;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: q.antialias, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, q.pixelRatio));
    this.renderer.shadowMap.enabled = q.shadows; this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.05;
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0xcfe1f2, span * 2, span * 14);

    this.sky = makeSky(); this.scene.add(this.sky.mesh);
    this.hemi = new THREE.HemisphereLight(0xdfeeff, 0x44503a, 0.85); this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff4e0, 2.2);
    this.sun.castShadow = q.shadows; this.sun.shadow.mapSize.set(q.shadowMap, q.shadowMap); this.sun.shadow.bias = -0.0004; this.sun.shadow.normalBias = 0.05;
    this.scene.add(this.sun); this.scene.add(this.sun.target);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 6000);
    this.fpvCam = new THREE.PerspectiveCamera(95, 1, 0.03, 6000);
    // FPV basis: camera -Z -> body +X (forward), camera +Y -> body +Z (up)
    this.fpvCam.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0)));
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08; this.controls.maxPolarAngle = Math.PI * 0.495;

    this.W = new THREE.Group(); this.W.rotation.x = -Math.PI / 2; this.scene.add(this.W); this.W.updateMatrixWorld(true);
    this.timeOfDay = 13; this.setTimeOfDay(13);
    this.setTheme(theme);
    this.resetCamera();
    this.frames = 0;
  }

  /** ENU position (array or Vector3) -> three.js world position. */
  toThree(v, out = new THREE.Vector3()) { out.set(v.x ?? v[0], v.y ?? v[1], v.z ?? v[2]); return this.W.localToWorld(out); }

  resetCamera() {
    const [cx, cy, cz] = this.center, d = Math.max(this.span * 1.5, 9);
    this.camera.position.copy(this.toThree([cx - d * 0.75, cy - d * 0.85, cz + d * 0.75]));
    this.controls.target.copy(this.toThree([cx, cy, cz])); this.controls.update();
  }

  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.fpvCam.aspect = w / h; this.fpvCam.updateProjectionMatrix();
  }

  setQuality(name) {
    const q = QUALITY[name] || QUALITY.high; this.quality = name;
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, q.pixelRatio));
    this.renderer.shadowMap.enabled = q.shadows; this.sun.castShadow = q.shadows;
    if (this.sun.shadow.mapSize.x !== q.shadowMap) { this.sun.shadow.mapSize.set(q.shadowMap, q.shadowMap); this.sun.shadow.map?.dispose(); this.sun.shadow.map = null; }
    this.scene.traverse((o) => { if (o.material) { const m = Array.isArray(o.material) ? o.material : [o.material]; m.forEach((mm) => { mm.needsUpdate = true; }); } });
  }

  /** hour in [0, 24): moves the sun, recolours sky, fog and lights. */
  setTimeOfDay(hour) {
    this.timeOfDay = hour;
    const st = applyTimeOfDay(hour, this);
    this.nightness = st.night;
    return st;
  }

  setTheme(theme) {
    this.theme = theme;
    if (theme === 'print') { this.scene.fog = null; this.sky.setPrint(true); this.renderer.setClearColor(0xffffff, 1); }
    else { this.sky.setPrint(false); this.scene.fog = this.scene.fog || new THREE.Fog(0xcfe1f2, this.span * 2, this.span * 14); this.setTimeOfDay(this.timeOfDay); }
  }

  /** Keep the shadow frustum tight around the focus so shadows stay sharp. */
  followSun(target3) {
    const dir = sunDirection(this.timeOfDay), s = Math.max(this.span * 0.9, 14);
    this.sun.position.copy(target3).addScaledVector(dir, this.span * 2.5);
    this.sun.target.position.copy(target3);
    const c = this.sun.shadow.camera;
    if (c.right !== s) { Object.assign(c, { left: -s, right: s, top: s, bottom: -s, near: 0.1, far: this.span * 8 }); c.updateProjectionMatrix(); }
  }

  render(cam = this.camera) { this.renderer.render(this.scene, cam); this.frames++; }

  dispose() { this.renderer.dispose(); }
}
