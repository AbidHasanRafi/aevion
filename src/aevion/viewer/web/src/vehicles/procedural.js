// A procedural multirotor built from the vehicle specification (arm layout, spin directions, rotor radius).

import * as THREE from 'three';

export const PALETTE = [0xff7a1a, 0x31a2ff, 0x39d98a, 0xd37cff, 0xffd23f, 0x4de8e0, 0xff5d8f, 0xb0b8c8];

export const DEFAULT_SPEC = { name: 'vehicle', mass: 1, rotors: [[0.2, 0, 0], [0, 0.2, 0], [-0.2, 0, 0], [0, -0.2, 0]], spin: [1, -1, 1, -1], radius: 0.1, w_max: 700 };

export function armLength(spec) { return Math.max(...spec.rotors.map((r) => Math.hypot(r[0], r[1])), 0.03); }

/** Visual scale factor that makes small vehicles visible at scene scale. */
export function autoScale(spec) { return Math.min(Math.max(0.32 / armLength(spec), 1), 9); }

export function buildDrone(spec, color, { light = false } = {}) {
  const root = new THREE.Group(), g = new THREE.Group(); root.add(g);
  const arm = armLength(spec), propR = spec.radius || arm * 0.5, L = arm;
  const bodyMat = new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.25 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1b2030, roughness: 0.6, metalness: 0.3 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.5 * L, 0.32 * L, 0.13 * L), bodyMat); body.castShadow = true; g.add(body);
  const top = new THREE.Mesh(new THREE.BoxGeometry(0.34 * L, 0.22 * L, 0.07 * L), dark); top.position.z = 0.09 * L; top.castShadow = true; g.add(top);
  const cam = new THREE.Mesh(new THREE.SphereGeometry(0.07 * L, 10, 8), new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.2, metalness: 0.6 })); cam.position.set(0.27 * L, 0, -0.02 * L); g.add(cam);
  const props = [];
  spec.rotors.forEach((r, i) => {
    const len = Math.hypot(r[0], r[1]), ang = Math.atan2(r[1], r[0]);
    const a = new THREE.Mesh(new THREE.CylinderGeometry(0.034 * L, 0.034 * L, len, 8), dark); a.rotation.z = ang - Math.PI / 2; a.position.set(r[0] / 2, r[1] / 2, 0); a.castShadow = true; g.add(a);
    const mot = new THREE.Mesh(new THREE.CylinderGeometry(0.085 * L, 0.085 * L, 0.11 * L, 14), dark); mot.rotation.x = Math.PI / 2; mot.position.set(r[0], r[1], (r[2] || 0) + 0.02 * L); mot.castShadow = true; g.add(mot);
    const led = new THREE.Mesh(new THREE.SphereGeometry(0.045 * L, 8, 6), new THREE.MeshBasicMaterial({ color: r[0] > 0 ? 0x39ff8a : 0xff4455 })); led.position.set(r[0], r[1], (r[2] || 0) - 0.06 * L); g.add(led);
    const pg = new THREE.Group(); pg.position.set(r[0], r[1], (r[2] || 0) + 0.1 * L);
    const bl = new THREE.MeshStandardMaterial({ color: 0xe9edf5, roughness: 0.5 });
    for (let k = 0; k < 2; k++) { const b = new THREE.Mesh(new THREE.BoxGeometry(2 * propR * 0.96, 0.09 * propR, 0.012 * propR), bl); b.rotation.z = k * Math.PI / 2; b.castShadow = true; pg.add(b); }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(propR, 28), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false })); disc.position.z = 0.004 * propR; pg.add(disc);
    g.add(pg); props.push({ g: pg, spin: spec.spin[i] || 1 });
  });
  const thrust = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1, 0x62f5ff, 0.16, 0.1); thrust.visible = false; g.add(thrust);
  root.userData = { g, props, thrust, body, arm, propR };
  return root;
}
