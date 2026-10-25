// Optional glTF/GLB vehicle model. The model replaces the procedural body; propellers are not animated for custom models.
//
// meta.vehicle.model = { url, scale?: 1, rotation?: [rx, ry, rz] (degrees), fit?: true }
// glTF is y-up with an arbitrary forward axis; Aevion bodies are x-forward, z-up (FLU). By default a +90 degree turn about x makes the model
// upright; `rotation` is applied after that to fix the heading. With fit (default) the model is scaled to the vehicle's arm span.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export async function loadVehicleModel(model, armSpan) {
  const gltf = await new GLTFLoader().loadAsync(model.url);
  const inner = gltf.scene, holder = new THREE.Group();
  inner.rotation.x = Math.PI / 2;                      // y-up -> z-up
  const rot = model.rotation || [0, 0, 0];
  holder.rotation.set(rot[0] * Math.PI / 180, rot[1] * Math.PI / 180, rot[2] * Math.PI / 180);
  holder.add(inner);
  holder.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(holder), size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
  const span = Math.max(size.x, size.y, 1e-6);
  const s = (model.scale ?? 1) * (model.fit === false ? 1 : (2 * armSpan) / span);
  holder.scale.setScalar(s);
  inner.position.sub(centre);                            // centre the model on the body origin
  holder.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = false; } });
  const root = new THREE.Group(), g = new THREE.Group(); root.add(g); g.add(holder);
  const thrust = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1, 0x62f5ff, 0.16, 0.1); thrust.visible = false; g.add(thrust);
  root.userData = { g, props: [], thrust, body: holder, arm: armSpan, propR: armSpan * 0.5, custom: true };
  return root;
}

/** Deep-ish clone suitable for placing several instances of one loaded model (materials are shared; geometry is shared). */
export function cloneVehicle(template) {
  const root = template.clone(true);
  const g = root.children[0], thrust = g.children.find((c) => c.isArrowHelper) || g.children[g.children.length - 1];
  root.userData = { ...template.userData, g, thrust, props: [] };
  return root;
}
