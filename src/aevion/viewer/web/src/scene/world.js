// World obstacles: boxes, spheres and vertical cylinders (trees get crowns). Instanced when there are many.

import * as THREE from 'three';
import { b64ToF32, b64ToBytes } from '../core/util.js';

const col = (arr, i, fb) => (arr && arr[i] ? new THREE.Color(arr[i][0], arr[i][1], arr[i][2]) : new THREE.Color(fb));
const INSTANCE_THRESHOLD = 24;

export function buildWorld(worldData, { theme = 'dark' } = {}) {
  const group = new THREE.Group();
  const stats = { boxes: 0, spheres: 0, cylinders: 0, instanced: false };
  if (!worldData) return { object: group, stats };
  const edgeOpacity = theme === 'print' ? 0.6 : 0.25;
  const dummy = new THREE.Object3D();

  // ---- boxes
  const boxes = worldData.boxes || [];
  stats.boxes = boxes.length;
  if (boxes.length > INSTANCE_THRESHOLD) {
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.05 }), boxes.length);
    boxes.forEach((b, i) => {
      dummy.position.set((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2); dummy.scale.set(b[3] - b[0], b[4] - b[1], b[5] - b[2]); dummy.rotation.set(0, 0, 0); dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix); mesh.setColorAt(i, col(worldData.box_colors, i, 0xbbbbbb));
    });
    mesh.castShadow = mesh.receiveShadow = true; mesh.instanceMatrix.needsUpdate = true; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    group.add(mesh); stats.instanced = true;
    // one merged edge geometry for all boxes (cheap): 12 edges each
    const seg = new Float32Array(boxes.length * 24 * 3); let o = 0;
    const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    for (const b of boxes) {
      const v = [[b[0], b[1], b[2]], [b[3], b[1], b[2]], [b[3], b[4], b[2]], [b[0], b[4], b[2]], [b[0], b[1], b[5]], [b[3], b[1], b[5]], [b[3], b[4], b[5]], [b[0], b[4], b[5]]];
      for (const [a, c] of E) { seg.set(v[a], o); seg.set(v[c], o + 3); o += 6; }
    }
    const eg = new THREE.BufferGeometry(); eg.setAttribute('position', new THREE.BufferAttribute(seg, 3));
    group.add(new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: edgeOpacity })));
  } else {
    boxes.forEach((b, i) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(b[3] - b[0], b[4] - b[1], b[5] - b[2]), new THREE.MeshStandardMaterial({ color: col(worldData.box_colors, i, 0xbbbbbb), roughness: 0.8, metalness: 0.05 }));
      m.position.set((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2); m.castShadow = m.receiveShadow = true; group.add(m);
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: edgeOpacity })); e.position.copy(m.position); group.add(e);
    });
  }

  // ---- spheres
  const spheres = worldData.spheres || [];
  stats.spheres = spheres.length;
  spheres.forEach((s, i) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(s[3], 28, 18), new THREE.MeshStandardMaterial({ color: col(worldData.sphere_colors, i, 0xcc8844), roughness: 0.6 }));
    m.position.set(s[0], s[1], s[2]); m.castShadow = m.receiveShadow = true; group.add(m);
  });

  // ---- cylinders (thin tall ones are trees: trunk + crown)
  const cyls = worldData.cylinders || [];
  stats.cylinders = cyls.length;
  if (cyls.length > INSTANCE_THRESHOLD) {
    const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1.08, 1, 14), new THREE.MeshStandardMaterial({ roughness: 0.9 }), cyls.length);
    const trees = cyls.map((c, i) => [c, i]).filter(([c]) => c[4] < 1.5 && c[3] - c[2] > 3);
    const crown = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 9), new THREE.MeshStandardMaterial({ color: new THREE.Color(0.22, 0.5, 0.2), roughness: 1 }), Math.max(trees.length, 1));
    cyls.forEach((c, i) => {
      const h = c[3] - c[2];
      dummy.position.set(c[0], c[1], c[2] + h / 2); dummy.rotation.set(Math.PI / 2, 0, 0); dummy.scale.set(c[4], h, c[4]); dummy.updateMatrix();
      trunk.setMatrixAt(i, dummy.matrix); trunk.setColorAt(i, col(worldData.cyl_colors, i, 0x6b8e4e));
    });
    trees.forEach(([c], k) => {
      dummy.position.set(c[0], c[1], c[3]); dummy.rotation.set(0, 0, 0); dummy.scale.set(c[4] * 3.2, c[4] * 3.2, c[4] * 3.2 * 0.8); dummy.updateMatrix(); crown.setMatrixAt(k, dummy.matrix);
    });
    crown.count = trees.length;
    for (const m of [trunk, crown]) { m.castShadow = true; m.receiveShadow = m === trunk; m.instanceMatrix.needsUpdate = true; group.add(m); }
    if (trunk.instanceColor) trunk.instanceColor.needsUpdate = true;
    stats.instanced = true;
  } else {
    cyls.forEach((c, i) => {
      const h = c[3] - c[2];
      const m = new THREE.Mesh(new THREE.CylinderGeometry(c[4], c[4] * 1.08, h, 20), new THREE.MeshStandardMaterial({ color: col(worldData.cyl_colors, i, 0x6b8e4e), roughness: 0.9 }));
      m.rotation.x = Math.PI / 2; m.position.set(c[0], c[1], c[2] + h / 2); m.castShadow = m.receiveShadow = true; group.add(m);
      if (c[4] < 1.5 && h > 3) {
        const crown = new THREE.Mesh(new THREE.SphereGeometry(c[4] * 3.2, 14, 10), new THREE.MeshStandardMaterial({ color: new THREE.Color(0.22, 0.5, 0.2), roughness: 1 }));
        crown.position.set(c[0], c[1], c[3]); crown.scale.z = 0.8; crown.castShadow = true; group.add(crown);
      }
    });
  }
  // ---- triangle meshes (imported models, map buildings): flat-shaded indexed geometry
  for (const m of worldData.meshes || []) {
    const geo = new THREE.BufferGeometry(), idx = b64ToBytes(m.faces);
    geo.setAttribute('position', new THREE.BufferAttribute(b64ToF32(m.vertices), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(idx.buffer, idx.byteOffset, idx.byteLength >> 2), 1));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: new THREE.Color(m.color[0], m.color[1], m.color[2]), roughness: 0.85, metalness: 0.05, flatShading: true }));
    mesh.castShadow = mesh.receiveShadow = true; group.add(mesh);
    stats.meshes = (stats.meshes || 0) + 1; stats.triangles = (stats.triangles || 0) + m.n_faces;
  }
  return { object: group, stats };
}
